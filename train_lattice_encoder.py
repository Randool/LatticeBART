"""
Train the encoder of LatticeBart in a contrastive-learning fashion
"""
import argparse
import glob
import json
import time
from logging import Logger
from math import ceil, floor
from os.path import exists
from typing import Union, List

import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed import ReduceOp
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, AdamW, get_cosine_schedule_with_warmup

from dataset.dataset import LatticeTextDataset, dict_to_device, DataProcessCfg, CollateFn
from models.config import get_model_config, get_tokenizer
from models.encoder import EncoderWrapper
from models.lattice_bart import LatticeBart
from models.model_utils import config_layer_wise_lr, model_init
from utils.args import *
from utils.metric import cos_similarity_mtx, contrastive_loss, show_similarity_mtx
from utils.optimizer import Lamb, Lookahead
from utils.utils import RuntimeMetric, get_logger, show_args, use_DDP, get_free_port, setup_dist_environment, set_seed, \
    use_DP, cleanup_dist_environment


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

    return args


def prepare_datasets(args: Namespace, logger: Logger):
    process_cfg = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=args.use_marginal, use_min_depth=args.use_min_depth,
        use_max_depth=args.use_max_depth, use_forward_backward=args.use_forward_backward,
        use_lattice_mask=args.use_lattice_mask, homophone_replace=args.homophone_replace,
        token_mask=args.token_mask, depth_disturb=args.depth_disturb,
    )
    process_cfg_for_test = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=1, use_min_depth=1, use_max_depth=1, use_forward_backward=1,
        use_lattice_mask=1, homophone_replace=0, token_mask=0, depth_disturb=0,
    )

    trainset, devset, debug_set, testsets = None, None, None, []

    if args.train_lat is not None or args.train_lat_ark is not None:
        trainset = LatticeTextDataset(lattice_pattern=args.train_lat,
                                      lattice_ark_pattern=args.train_lat_ark,
                                      process_cfg=process_cfg)

    if args.eval_lat is not None or args.eval_lat_ark is not None:
        devset = LatticeTextDataset(lattice_pattern=args.eval_lat,
                                    lattice_ark_pattern=args.eval_lat_ark,
                                    process_cfg=process_cfg_for_test)

    if args.debug_lat is not None or args.debug_lat_ark is not None:
        debug_set = LatticeTextDataset(lattice_pattern=args.debug_lat,
                                       lattice_ark_pattern=args.debug_lat_ark,
                                       process_cfg=process_cfg_for_test)

    if len(args.test_lat) == 0 and len(args.test_lat_ark) == 0:
        logger.error("No test data found.")

    if len(args.test_lat) > 0 and len(args.test_lat_ark) > 0:
        logger.error(f"Only one `--test_lat` and `--test_lat_ark` of can be set. "
                     f"Got: test_lat: {args.test_lat}, test_lat_ark: {args.test_lat_ark}")

    test_lat = [None] * len(args.test_lat_ark) if len(args.test_lat) == 0 else args.test_lat
    test_lat_ark = [None] * len(args.test_lat) if len(args.test_lat_ark) == 0 else args.test_lat_ark

    for i, (_lat, _lat_ark) in enumerate(zip(test_lat, test_lat_ark)):
        testset = LatticeTextDataset(lattice_pattern=_lat, lattice_ark_pattern=_lat_ark,
                                     process_cfg=process_cfg_for_test)
        testsets.append(testset)

    return trainset, devset, debug_set, testsets


def select_best_checkpoint(exp_dir: str, use_metric: str, logger: Logger = None):
    metrics = glob.glob(join(exp_dir, "epoch_*", "metric_*"))
    if len(metrics) == 0:
        return None

    best_exp_dir, best_metric, largest_n = None, None, -1

    for m in metrics:
        *prefix, metric = m.split("/")
        prefix = "/".join(prefix)
        if metric[-1] == "%":  # wer cer
            metric = float(metric.split("_")[-1][:-1])
        else:
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
                if best_metric < metric or (best_metric == metrics and n > largest_n):
                    best_metric = metric
                    best_exp_dir = prefix
                    largest_n = n

    if logger:
        logger.info(f"Select best ckpt: {best_exp_dir}")

    return best_exp_dir


def get_encoder_and_tokenizer(args: Namespace, logger: Logger, to_device):
    ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric, logger=logger)
    if ckpt_dir is not None:
        encoder = EncoderWrapper.from_pretrained(ckpt_dir)
        tokenizer = BartTokenizer.from_pretrained(args.exp_dir)
        if logger:
            logger.info(f"Load checkpoint from '{ckpt_dir}'")
    else:
        if args.default_config is not None:
            config = get_model_config(args.default_config)
            encoder = LatticeBart(config).get_encoder()
            tokenizer = get_tokenizer()

        elif args.bart_path is not None:
            encoder = LatticeBart.from_bart(bart_path=args.bart_path, lat_enc_layers=args.lat_enc_layers).get_encoder()
            tokenizer = BartTokenizer.from_pretrained(args.bart_path)

        elif args.lat_bart_path is not None:
            encoder = LatticeBart.from_pretrained(args.lat_bart_path).get_encoder()
            tokenizer = BartTokenizer.from_pretrained(args.lat_bart_path)

        else:
            if logger:
                logger.error("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")
            raise ValueError("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")

        if args.init_model:
            if logger:
                logger.info("[!] Initialize model parameters.")
            model_init(encoder, std=encoder.config.init_std, except_modules=(encoder.embed_tokens,))

        encoder = EncoderWrapper(encoder)

    encoder.to(to_device)
    return encoder, tokenizer


def wrap_distributed_parallel(rank, model: Union[EncoderWrapper, LatticeBart], args: Namespace, logger: Logger):
    if args.parallel == "DP" and args.ngpus > 1:
        if logger:
            logger.info(f"Found {args.ngpus} GPUs, using DataParallel")
        model = DP(model.to(args.device))
    elif args.parallel == "DDP" and args.ngpus > 1:
        if logger:
            logger.info(f"Rank: {rank}. Found {args.ngpus} GPUs, using DistributedDataParallel")
        torch.cuda.set_device(f"cuda:{rank}")
        model = DDP(model.to(f"cuda:{rank}"), device_ids=[rank], output_device=0)
    return model


def config_optimizer(encoder: Union[EncoderWrapper, LatticeBart], args: Namespace):
    name = args.optim.lower()
    param_list = config_layer_wise_lr(encoder, args.lr, args.lr_decay)
    if name == "sgd":
        optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif name == "adam":
        optim = AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    elif name == "lamb":
        optim = Lamb(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9, adam=True)
    elif name == "lookahead":
        base_optim = AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
        optim = Lookahead(base_optim)
    else:
        raise ValueError(f"Unsupported optimizer called {args.optim}")
    return optim


def save_checkpoint(epoch: int, model, metric, optimizer, tokenizer: BartTokenizer, runtime_metric: RuntimeMetric,
                    args: Namespace, logger: Logger):
    exp_dir = join(args.exp_dir, f"epoch_{epoch}")
    os.makedirs(exp_dir, exist_ok=True)

    # Model state and config.json
    if isinstance(model, (DP, DDP)):
        model.module.save_pretrained(exp_dir)
    else:
        model.save_pretrained(exp_dir)

    # Save metric
    if args.use_metric in ("loss",):
        open(join(exp_dir, f"metric_{metric:.2e}"), "w")
    elif args.use_metric in ("wer", "cer"):
        open(join(exp_dir, f"metric_{metric:.2%}"), "w")
    else:  # "bleu"
        open(join(exp_dir, f"metric_{metric:.2f}"), "w")

    # # Optimizer state
    torch.save(optimizer.state_dict(), join(exp_dir, "optim.pt"))
    # Tokenizer
    tokenizer.save_pretrained(args.exp_dir)

    # Metric
    runtime_metric.dump(join(exp_dir, "runtime_metric.json"))

    if logger:
        logger.info(f"Save checkpoints to {exp_dir}")


def evaluate(encoder: EncoderWrapper, tokenizer: BartTokenizer, dataloader: DataLoader, args: Namespace,
             logger: Logger) -> dict:
    _state = encoder.training
    encoder.eval()
    device = encoder.device if isinstance(encoder, EncoderWrapper) else encoder.module.device
    total_loss = 0
    eos_id = tokenizer.eos_token_id
    pool = args.pooling_method
    weight = None
    if args.token_id_weight:
        with open(args.token_id_weight, encoding="utf8") as f:
            _weight = json.load(f)
            weight = {int(key): val for key, val in _weight.items()}

    with torch.no_grad():
        for inputs in dataloader:
            inputs = dict_to_device(inputs, device)
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
    if logger:
        logger.info(f"Eval | loss: {avg_loss:.2e}")
    else:
        print(f"Eval | loss: {avg_loss:.2e}")
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


def training(encoder: Union[EncoderWrapper, DP, DDP], tokenizer: BartTokenizer, train_loader: DataLoader,
             dev_loader: DataLoader, optimizer, scheduler, writer: SummaryWriter, args: Namespace, logger: Logger,
             epoch_trained=1, runtime_metric: RuntimeMetric = None, debug_loader: DataLoader = None):
    is_DP, is_DDP = isinstance(encoder, DP), isinstance(encoder, DDP)
    device = encoder.module.device if (is_DP or is_DDP) else encoder.device
    early_stop = torch.tensor(0).to("cuda" if is_DDP else "cpu")

    if is_DDP and dist.get_rank() != 0:
        mtc = None
    else:
        # Metrics
        mtc = RuntimeMetric(
            None, window_size=10, patience=args.patience, decrease=(args.use_metric in ("cer", "wer", 'loss'))
        ) if runtime_metric is None else runtime_metric

    if logger:
        logger.info("======== Start training ========")

    eos_id = tokenizer.eos_token_id
    pool = args.pooling_method
    weight = None
    if args.token_id_weight:
        with open(args.token_id_weight, encoding="utf8") as f:
            _weight = json.load(f)
            weight = {int(key): val for key, val in _weight.items()}

    for epoch in range(epoch_trained, args.epoch + 1):
        encoder.train()

        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        max_steps = len(train_loader)
        summary_every = max_steps // 10
        tic = time.time()

        for step, inputs in enumerate(train_loader, start=1):
            inputs = dict_to_device(inputs, device)

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
            if mtc:
                mtc.avg_loss.append(loss.item())

            global_step = len(train_loader) * (epoch - 1) + step
            if writer:
                writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step)

            if logger and step % args.log_every == 0:
                lrs = [f"{group['lr']:.2e}" for group in optimizer.param_groups]
                lrs = [lrs[0], lrs[-1]] if len(lrs) > 2 else lrs
                lrs = " ~ ".join(lrs)
                logger.info(f"Ep {epoch} / {args.epoch} | st {step} / {max_steps} | lr: {lrs} | "
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
            if writer and debug_loader and step % summary_every == 0:
                show_similarity_matrix(encoder, tokenizer, debug_loader, args, writer, global_step)

        if logger:
            logger.info(f"Epoch {epoch} training finish. Cost {(time.time() - tic) / 60:.1f} min")

        # Evaluate
        if writer:
            metrics = evaluate(encoder, tokenizer, dev_loader, args, logger=logger)
            writer.add_scalars(
                main_tag="Metric", tag_scalar_dict=metrics,
                global_step=len(train_loader) * epoch
            )
            metric = metrics[args.use_metric]

            if mtc.metric_is_better_than_ever(metric) or args.always_save:
                save_checkpoint(epoch=epoch, model=encoder, metric=metric, optimizer=optimizer,
                                tokenizer=tokenizer, runtime_metric=mtc, args=args, logger=logger)

        # If meet the requirement of early_stop (then inform other processes), then break the loop
        if mtc and mtc.do_early_stop:
            early_stop.fill_(1)

        if is_DDP:
            dist.all_reduce(early_stop, ReduceOp.SUM)

        if early_stop > 0:
            if logger:
                logger.info(f"Early stopping after {epoch} epoch.")
            break


def train(rank: int, world_size: int, master_port: int, trainset: LatticeTextDataset, devset: LatticeTextDataset,
          args: Namespace, debugset: LatticeTextDataset = None, logger: Logger = None):
    if trainset is None or len(trainset) == 0:
        logger.error("Empty trainset")
        return

    if devset is None or len(devset) == 0:
        logger.error("Empty devset")
        return

    if rank != -1:
        setup_dist_environment(rank, world_size, master_port)

    set_seed(args.seed)

    if logger is None and rank == 0:
        logger = get_logger(args, "LatticeEncoderTrain")
    writer = SummaryWriter(log_dir=join(args.exp_dir, "tensorboard")) if rank <= 0 else None

    _use_DP, _use_DDP = use_DP(args), use_DDP(args)

    if _use_DDP:
        device = f"cuda:{rank}"
        encoder, tokenizer = get_encoder_and_tokenizer(args, logger, to_device=device)
        batch_size = args.train_batch // world_size
        num_workers = args.num_workers // world_size
    else:
        device = args.device
        encoder, tokenizer = get_encoder_and_tokenizer(args, logger, to_device=device)
        batch_size = args.train_batch
        num_workers = args.num_workers

    collate_fn = CollateFn(tokenizer=tokenizer, process_cfg=trainset.process_cfg, decoder_input=args.decoder_input,
                           n_lat=(2 if args.contrast_method == "mask" else 1)).collate_fn

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False if _use_DDP else True, num_workers=num_workers,
        collate_fn=collate_fn, sampler=DistributedSampler(trainset, seed=args.seed) if _use_DDP else None
    )
    dev_loader = DataLoader(
        devset, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )
    if debugset is None:
        debug_loader = None
    else:
        assert abs(len(debugset) // debugset.n_files * debugset.n_files - len(debugset)) <= 1e-4, \
            "Debug dataset files have different lattice number."
        debug_loader = DataLoader(
            debugset, batch_size=len(debugset) // debugset.n_files, shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn
        )

    optimizer = config_optimizer(encoder, args)
    total_steps = ceil(args.epoch * len(trainset) / (args.train_batch * args.grad_acc_steps))
    if logger:
        logger.info(f"Total step: {total_steps}, warm-up steps: {floor(total_steps * 0.05)}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=floor(total_steps * 0.01),
        num_training_steps=floor(total_steps * 1.2)  # Prevent zero lr
    )

    mtc = RuntimeMetric(None, window_size=10, patience=args.patience,
                        decrease=(args.use_metric in ("cer", "wer", "loss")))
    epoch_trained = 1

    ### Fast Forward ###
    ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric, logger=logger)
    if ckpt_dir is not None:
        epoch_trained = int(re.match(r".*epoch_(?P<n>\d+).*", ckpt_dir).group("n"))
        step_trained = ceil(epoch_trained * len(trainset) / (args.train_batch * args.grad_acc_steps))

        optimizer.load_state_dict(torch.load(join(ckpt_dir, "optim.pt"), map_location=args.device))

        for _ in range(step_trained):
            scheduler.step()

        for i in range(epoch_trained):
            for _ in train_loader:
                break
            if logger:
                logger.info(f"Pass epoch {i + 1}")

        mtc = RuntimeMetric.load(join(ckpt_dir, "runtime_metric.json"))

    if _use_DP or _use_DDP:
        encoder = wrap_distributed_parallel(rank=rank, model=encoder, args=args, logger=logger)

    try:
        training(encoder=encoder, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader,
                 optimizer=optimizer, scheduler=scheduler, writer=writer, args=args, logger=logger,
                 epoch_trained=epoch_trained, runtime_metric=mtc, debug_loader=debug_loader)
    except KeyboardInterrupt:
        if logger:
            logger.info("Interrupt by user.")

    if rank != -1:
        cleanup_dist_environment()


def test(testsets: List[LatticeTextDataset], args: Namespace, logger: Logger):
    if len(testsets) == 0:
        logger.error("No test data found.")
        return
    set_seed(args.seed)

    exp_dir = select_best_checkpoint(args.exp_dir, args.use_metric, logger=logger)
    if exp_dir is None:
        logger.warning("No checkpoints found.")
        return

    try:
        encoder = EncoderWrapper.from_pretrained(exp_dir).to(args.device).eval()
        tokenizer = BartTokenizer.from_pretrained(args.exp_dir)

        if use_DP(args):
            encoder = wrap_distributed_parallel(rank=-1, model=encoder, args=args, logger=logger)

        if logger:
            logger.info("======== Start testing ========")

        for i, testset in enumerate(testsets):
            collate_fn = CollateFn(tokenizer=tokenizer, process_cfg=testset.process_cfg,
                                   decoder_input=args.decoder_input,
                                   n_lat=(2 if args.contrast_method == "mask" else 1)).collate_fn
            test_loader = DataLoader(testset, batch_size=args.eval_batch, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn)
            evaluate(encoder=encoder, tokenizer=tokenizer, dataloader=test_loader, args=args, logger=logger)

    except FileNotFoundError:
        if logger:
            logger.error("FileNotFoundError!")

        config_file = join(exp_dir, "config.json")
        enc_pt_file = join(exp_dir, EncoderWrapper.ENC_PT_FILE)
        proj_pt_file = join(exp_dir, EncoderWrapper.PROJ_PT_FILE)

        if logger:
            if not exists(config_file):
                logger.error(f"config_file: `{config_file}` not found")
            if not exists(enc_pt_file):
                logger.error(f"enc_pt_file: `{enc_pt_file}` not found")
            if not exists(proj_pt_file):
                logger.error(f"proj_pt_file: `{proj_pt_file}` not found")


def main(_args: List[str] = None):
    args = parse_args_and_setup_hyper_param(_args)
    os.makedirs(args.exp_dir, exist_ok=True)

    logger = get_logger(args, "LatticeEncoderMain")
    show_args(args, logger)

    trainset, devset, debug_set, testsets = prepare_datasets(args, logger)
    train_phase = test_phase = True

    if trainset is None or len(trainset) == 0:
        logger.info("Trainset: \t0 items")
        train_phase = False
    else:
        logger.info(f"Trainset: \t{len(trainset)} items")

    if devset is None or len(devset) == 0:
        logger.info("Devset:\t0 items")
        train_phase = False
    else:
        logger.info(f"Devset: \t{len(devset)} items")

    if len(testsets) == 0:
        logger.info("Testset: \t0 items")
        test_phase = False
    else:
        for i, testset in enumerate(testsets):
            logger.info(f"Testset {i}: \t{len(testset)} items")

    if train_phase:
        if use_DDP(args):
            world_size = torch.cuda.device_count()
            logger.info(f"word_size: {world_size}")
            port = get_free_port()
            mp.spawn(train, args=(world_size, port, trainset, devset, args, None), nprocs=world_size, join=True)
        else:
            train(rank=-1, world_size=-1, master_port=-1, trainset=trainset, devset=devset, args=args,
                  debugset=debug_set, logger=logger)
    else:
        logger.info("Training phase is passed.")

    if test_phase:
        test(testsets=testsets, args=args, logger=logger)
    else:
        logger.info("Testing phase is passed.")

    return args.exp_dir


if __name__ == '__main__':
    main()
