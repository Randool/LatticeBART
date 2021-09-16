"""
Train the encoder of LatticeBart in a contrastive-learning fashion
"""
import argparse
import glob
import json
import logging
import time
from math import ceil, floor
from os.path import exists
from typing import Union, List

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, AdamW, get_cosine_schedule_with_warmup

from dataset.dataset import LatticeTextDataset, dict_to_device, DataProcessCfg
from models.config import get_model_config, get_tokenizer
from models.encoder import EncoderWrapper
from models.lattice_bart import LatticeBart
from models.utils import config_layer_wise_lr, model_init
from utils.args import *
from utils.lamb import Lamb
from utils.metric import cos_similarity_mtx, contrastive_loss, show_similarity_mtx
from utils.utils import set_seed, set_logger, RuntimeMetric


def parse_args_and_setup_hyper_param(_args: List[str] = None) -> Namespace:
    psr = argparse.ArgumentParser()
    add_args_for_model_construct(psr)
    add_args_for_lr(psr)
    add_args_for_data_path(psr)
    add_args_for_data_process(psr)
    add_args_for_train_procedure(psr)
    add_args_for_contrastive_learning(psr)
    args = psr.parse_args(_args)

    for key, val in list(vars(args).items()):
        if val is None:
            continue
        if isinstance(val, str):
            setattr(args, key, val.strip())

    config_exp_dir(args)

    set_seed(args.seed)
    set_logger(args)

    return args


def select_best_checkpoint(exp_dir: str, use_metric: str):
    metrics = glob.glob(join(exp_dir, "epoch_*", "metric_*"))
    if len(metrics) == 0:
        return None

    best_exp_dir, best_metric, largest_n = None, None, -1

    for m in metrics:
        *prefix, metric = m.split("/")
        prefix = "/".join(prefix)
        metric = float(metric.split("_")[-1])
        n = int(re.match(r".*epoch_(?P<n>\d+).*", prefix).group("n"))

        if best_metric is None:
            best_metric = metric
            largest_n = n
            best_exp_dir = prefix
        else:
            if use_metric in ("cer", "wer", "loss"):
                if best_metric > metric or (best_metric == metrics and n > largest_n):
                    best_metric = metric
                    best_exp_dir = prefix
                    largest_n = n
            else:
                if best_metric < metric and (best_metric == metrics and n > largest_n):
                    best_metric = metric
                    best_exp_dir = prefix
                    largest_n = n

    logging.info(f"Select checkpoint {best_exp_dir}")

    return best_exp_dir


def get_encoder_and_tokenizer(args: Namespace):
    ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric)
    if ckpt_dir is not None:
        encoder = EncoderWrapper.from_pretrained(ckpt_dir)
        tokenizer = BartTokenizer.from_pretrained(args.exp_dir)
        logging.info(f"Load checkpoint from '{ckpt_dir}'")
    else:
        if args.default_config is not None:
            encoder = LatticeBart(get_model_config(args.default_config)).get_encoder()
            tokenizer = get_tokenizer()
        elif args.bart_path is not None:
            encoder = LatticeBart.from_bart(bart_path=args.bart_path, depth_clip=args.depth_clip,
                                            lat_enc_layers=args.lat_enc_layers).get_encoder()
            tokenizer = BartTokenizer.from_pretrained(args.bart_path)
        elif args.lat_bart_path is not None:
            encoder = LatticeBart.from_pretrained(args.lat_bart_path).get_encoder()
            tokenizer = BartTokenizer.from_pretrained(args.lat_bart_path)
        else:
            logging.error("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")
            raise ValueError("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")

        if args.init_model:
            logging.info("[!] Initialize model parameters.")
            model_init(encoder, std=encoder.config.init_std, except_modules=(encoder.embed_tokens,))

        encoder = EncoderWrapper(encoder)

    # If you have multiple GPUs, why not use them?
    if "cuda" in args.device and torch.cuda.device_count() > 1:
        logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed training")
        encoder = nn.DataParallel(encoder)

    return encoder, tokenizer


def config_optimizer(encoder: Union[EncoderWrapper, LatticeBart], args: Namespace):
    name = args.optim.lower()
    param_list = config_layer_wise_lr(encoder, args.lr, args.lr_decay)
    if name == "sgd":
        optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif name == "adam":
        optim = AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    elif name == "lamb":
        optim = Lamb(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9, adam=True)
    else:
        logging.error(f"Unsupported optimizer called {args.optim}")
        raise ValueError(f"Unsupported optimizer called {args.optim}")
    return optim


def save_checkpoint(epoch: int, metric, model, optimizer, tokenizer: BartTokenizer, runtime_metric: RuntimeMetric,
                    args: Namespace):
    exp_dir = join(args.exp_dir, f"epoch_{epoch}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save metric
    if args.use_metric in ("loss",):
        open(join(exp_dir, f"metric_{metric:.2e}"), "w")
    elif args.use_metric in ("wer", "cer"):
        open(join(exp_dir, f"metric_{metric:.2%}"), "w")
    else:  # "bleu"
        open(join(exp_dir, f"metric_{metric:.2f}"), "w")

    # Model state and config.json
    model.save_pretrained(exp_dir)
    # # Optimizer state
    torch.save(optimizer.state_dict(), join(exp_dir, "optim.pt"))
    # Tokenizer
    tokenizer.save_pretrained(args.exp_dir)

    # Metric
    runtime_metric.dump(join(exp_dir, "runtime_metric.json"))

    logging.info(f"Save checkpoints to {exp_dir}")


def evaluate(encoder: EncoderWrapper, tokenizer: BartTokenizer, dataloader: DataLoader, args: Namespace) -> dict:
    _state = encoder.training
    encoder.eval()
    total_loss = 0
    eos_id = tokenizer.eos_token_id
    pool = args.pooling_method
    weight = None
    if args.token_id_weight:
        with open(args.token_id_weight, encoding="utf8") as f:
            _weight = json.load(f)
            weight = {int(key): val for key, val in _weight.items()}

    for inputs in dataloader:
        inputs = dict_to_device(inputs, args.device)
        with torch.no_grad():
            if args.contrast_method == "dropout":
                last_state1 = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
                last_state2 = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
            else:
                last_state = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
                last_state1 = last_state[::2]
                last_state2 = last_state[1::2]

            sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
            total_loss += contrastive_loss(sim_mtx).item()

    if _state:
        encoder.train()

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Eval | loss: {avg_loss:.2e}")
    return {"loss": avg_loss}


def show_similarity_matrix(encoder: EncoderWrapper, tokenizer: BartTokenizer, dataloader: DataLoader, args: Namespace,
                           writer: SummaryWriter, global_step: int):
    _state = encoder.training
    encoder.eval()

    total_loss = 0
    eos_id = tokenizer.eos_token_id
    pool = args.pooling_method
    weight = None
    if args.token_id_weight:
        with open(args.token_id_weight, encoding="utf8") as f:
            _weight = json.load(f)
            weight = {int(key): val for key, val in _weight.items()}

    for i, inputs in enumerate(dataloader):
        key = inputs.pop("key", None)
        if key is None:
            key = [i for i in range(len(inputs["input_ids"]))]

        with torch.no_grad():
            inputs = dict_to_device(inputs, args.device)
            last_state = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
            last_state1 = last_state[::2]
            last_state2 = last_state[1::2]

            sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
            total_loss += contrastive_loss(sim_mtx).item()

            fig = show_similarity_mtx(sim_mtx, include_values=False, display_labels=key)
            writer.add_figure(tag=f"SimilarityMtx_{i}", figure=fig, global_step=global_step)

    if _state:
        encoder.train()

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar(tag="Similarity Mtx Loss", scalar_value=avg_loss, global_step=global_step)


def train(encoder: Union[EncoderWrapper, nn.DataParallel], tokenizer: BartTokenizer, train_loader: DataLoader,
          dev_loader: DataLoader, optimizer, scheduler, writer: SummaryWriter, args: Namespace, epoch_trained=1,
          runtime_metric: RuntimeMetric = None, debug_loader: DataLoader = None):
    # Model
    encoder.to(args.device)

    # Metrics
    mtc = RuntimeMetric(None, window_size=10, patience=args.patience,
                        decrease=(args.use_metric in ("cer", "wer", 'loss'))) if runtime_metric is None \
        else runtime_metric

    logging.info("==== Start training ====")
    eos_id = tokenizer.eos_token_id
    pool = args.pooling_method
    weight = None
    if args.token_id_weight:
        with open(args.token_id_weight, encoding="utf8") as f:
            _weight = json.load(f)
            weight = {int(key): val for key, val in _weight.items()}

    for epoch in range(epoch_trained, args.epoch + 1):
        encoder.train()

        max_steps = len(train_loader)
        summary_every = max_steps // 10
        tic = time.time()

        for step, inputs in enumerate(train_loader, start=1):
            inputs = dict_to_device(inputs, args.device)

            if args.contrast_method == "dropout":
                last_state1 = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
                last_state2 = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
            else:  # mask
                last_state = encoder.forward(**inputs, eos_id=eos_id, pooling_method=pool, token_id_weight=weight)
                last_state1 = last_state[::2]
                last_state2 = last_state[1::2]

            sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
            loss = contrastive_loss(sim_mtx, reduction="mean")
            loss.backward()
            mtc.avg_loss.append(loss.item())

            global_step = len(train_loader) * (epoch - 1) + step
            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step)

            if step % args.log_every == 0:
                lrs = [f"{group['lr']:.2e}" for group in optimizer.param_groups]
                lrs = [lrs[0], lrs[-1]] if len(lrs) > 2 else lrs
                lrs = " ~ ".join(lrs)
                logging.info(f"Ep {epoch} / {args.epoch} | st {step} / {max_steps} | lr: {lrs} | "
                             f"avg loss: {mtc.avg_loss.avg_value:.2e}")

            if global_step % args.grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
                # # 监视梯度
                # if global_step % (args.grad_acc_steps * 50) == 0:
                #     for name, param in encoder.named_parameters():
                #         writer.add_histogram(tag=f"grad_{name}", values=param.grad.data.cpu().numpy(),
                #                              global_step=global_step)
                #         writer.add_histogram(tag=f"param_{name}", values=param.data.cpu().numpy(),
                #                              global_step=global_step)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Show similarity matrix for manually selected lattices
            if step % summary_every == 0:
                if debug_loader is not None:
                    show_similarity_matrix(encoder, tokenizer, debug_loader, args, writer, global_step)

        span = (time.time() - tic) / 60
        logging.info(f"Epoch {epoch} training finish. Cost {span:.1f} min")

        # Evaluate
        metrics = evaluate(encoder, tokenizer, dev_loader, args)
        writer.add_scalars(
            main_tag="Metric", tag_scalar_dict=metrics,
            global_step=len(train_loader) * epoch
        )
        metric = metrics[args.use_metric]

        if mtc.best_metrics is None or \
                (mtc.best_metrics < metric and args.use_metric in ("bleu",)) or \
                (mtc.best_metrics > metric and args.use_metric in ("cer", "wer", "loss")):
            mtc.best_metrics = metric

            _m = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
            save_checkpoint(epoch=epoch, metric=metric, model=_m, optimizer=optimizer,
                            tokenizer=tokenizer, runtime_metric=mtc, args=args)

        if mtc.early_stop.stop(metric):
            logging.info(f"Early stopping after {epoch} epoch.")
            break


def main(_args: List[str] = None):
    args = parse_args_and_setup_hyper_param(_args)

    #########################
    ######## Dataset ########
    #########################
    process_cfg = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=args.use_marginal, use_min_depth=args.use_min_depth,
        use_max_depth=args.use_max_depth, use_forward_backward=args.use_forward_backward,
        use_lattice_mask=args.use_lattice_mask, binary_mask=args.use_binary_mask,
        homophone_replace=args.homophone_replace
    )
    process_cfg_for_test = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=1, use_min_depth=1, use_max_depth=1, use_forward_backward=1,
        use_lattice_mask=1, binary_mask=1, homophone_replace=0
    )

    trainset = devset = debug_set = None

    if args.train_lat is not None or args.train_lat_ark is not None:
        trainset = LatticeTextDataset(lattice_pattern=args.train_lat,
                                      lattice_ark_pattern=args.train_lat_ark,
                                      cache_dir=join(args.cache_dir, "cache_train"),
                                      process_cfg=process_cfg)

    if args.eval_lat is not None or args.eval_lat_ark is not None:
        devset = LatticeTextDataset(lattice_pattern=args.eval_lat,
                                    lattice_ark_pattern=args.eval_lat_ark,
                                    cache_dir=join(args.cache_dir, "cache_dev"),
                                    process_cfg=process_cfg_for_test)

    if args.debug_lat is not None or args.debug_lat_ark is not None:
        debug_set = LatticeTextDataset(lattice_pattern=args.debug_lat,
                                       lattice_ark_pattern=args.debug_lat_ark,
                                       cache_dir=join(args.cache_dir, "cache_dev"),
                                       process_cfg=process_cfg_for_test)

    writer = SummaryWriter(log_dir=join(args.exp_dir, f"tensorboard_{time.strftime('%Y-%m-%d_%H-%M')}"))

    #######################
    ######## Train ########
    #######################
    if trainset is None or len(trainset) == 0:
        logging.info("No valid training data. Training phase is passed.")
    elif devset is None or len(devset) == 0:
        logging.info("No valid validation data. Training phase is passed.")
    else:
        encoder, tokenizer = get_encoder_and_tokenizer(args)

        collate_fn = trainset.partial_collate_fn(tokenizer, n_lat=(2 if args.contrast_method == "mask" else 1))

        train_loader = DataLoader(
            trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers,
            collate_fn=collate_fn, drop_last=isinstance(encoder, nn.DataParallel)
        )
        dev_loader = DataLoader(
            devset, batch_size=args.eval_batch, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn
        )
        assert abs(len(debug_set) // debug_set.n_files * debug_set.n_files - len(debug_set)) <= 1e-4, \
            "Debug dataset files have different lattice number."
        debug_loader = DataLoader(
            debug_set, batch_size=len(debug_set) // debug_set.n_files, shuffle=False, num_workers=args.num_workers,
            collate_fn=trainset.partial_collate_fn(tokenizer, n_lat=2, return_keys=True)
        ) if debug_set is not None else None

        optimizer = config_optimizer(encoder, args)
        total_steps = ceil(args.epoch * len(trainset) / (args.train_batch * args.grad_acc_steps))
        logging.info(f"Total step: {total_steps}, warm-up steps: {floor(total_steps * 0.05)}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=floor(total_steps * 0.01),
            num_training_steps=total_steps
        )

        mtc = RuntimeMetric(None, window_size=10, patience=args.patience,
                            decrease=(args.use_metric in ("cer", "wer", "loss")))
        epoch_trained = 1

        ####################
        ### Fast Forward ###
        ####################
        ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric)
        if ckpt_dir is not None:
            epoch_trained = int(re.match(r".*epoch_(?P<n>\d+).*", ckpt_dir).group("n"))
            step_trained = ceil(epoch_trained * len(trainset) / (args.train_batch * args.grad_acc_steps))

            optimizer.load_state_dict(torch.load(join(ckpt_dir, "optim.pt"), map_location=args.device))

            for _ in range(step_trained):
                scheduler.step()

            for i in range(epoch_trained):
                for _ in train_loader:
                    break
                logging.info(f"Pass epoch {i + 1}")

            mtc = RuntimeMetric.load(join(ckpt_dir, "runtime_metric.json"))

        try:
            train(encoder=encoder, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader,
                  optimizer=optimizer, scheduler=scheduler, writer=writer, args=args, epoch_trained=epoch_trained,
                  runtime_metric=mtc, debug_loader=debug_loader)
        except KeyboardInterrupt:
            logging.info("Interrupt by user.")

    ######## Test ########
    # Gather metrics
    exp_dir = select_best_checkpoint(args.exp_dir, args.use_metric)
    if exp_dir is None:
        logging.warning("No checkpoints found.")
    else:
        if len(args.test_lat) == 0 and len(args.test_lat_ark) == 0:
            logging.error("No test data found.")
            return
        if len(args.test_lat) > 0 and len(args.test_lat_ark) > 0:
            logging.error(f"Only one `--test_lat` and `--test_lat_ark` of can be set. "
                          f"Got: test_lat: {args.test_lat}, test_lat_ark: {args.test_lat_ark}")
            return
        try:
            encoder = EncoderWrapper.from_pretrained(exp_dir).to(args.device).eval()
            tokenizer = BartTokenizer.from_pretrained(args.exp_dir)

            if "cuda" in args.device and torch.cuda.device_count() > 1:
                logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed testing")
                encoder = nn.DataParallel(encoder)

            logging.info("==== Start testing ====")

            cache_test = join(args.cache_dir, "cache_test")
            test_lat = [None] * len(args.test_lat_ark) if len(args.test_lat) == 0 else args.test_lat
            test_lat_ark = [None] * len(args.test_lat) if len(args.test_lat_ark) == 0 else args.test_lat_ark

            for _lat, _lat_ark in zip(test_lat, test_lat_ark):
                testset = LatticeTextDataset(lattice_pattern=_lat, lattice_ark_pattern=_lat_ark,
                                             text_pattern=args.test_text, cache_dir=cache_test,
                                             process_cfg=process_cfg_for_test)
                collate_fn = testset.partial_collate_fn(tokenizer, n_lat=(2 if args.contrast_method == "mask" else 1))
                test_loader = DataLoader(testset, batch_size=args.eval_batch, shuffle=False,
                                         num_workers=args.num_workers, collate_fn=collate_fn)
                evaluate(encoder=encoder, tokenizer=tokenizer, dataloader=test_loader, args=args)

        except FileNotFoundError:
            logging.error("FileNotFoundError!")

            config_file = join(exp_dir, "config.json")
            enc_pt_file = join(exp_dir, EncoderWrapper.ENC_PT_FILE)
            proj_pt_file = join(exp_dir, EncoderWrapper.PROJ_PT_FILE)

            if not exists(config_file):
                logging.error(f"config_file: `{config_file}` not found")
            if not exists(enc_pt_file):
                logging.error(f"enc_pt_file: `{enc_pt_file}` not found")
            if not exists(proj_pt_file):
                logging.error(f"proj_pt_file: `{proj_pt_file}` not found")


if __name__ == '__main__':
    main()
