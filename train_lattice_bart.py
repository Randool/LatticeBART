"""
Training pipeline of LatticeBart
"""
import argparse
from logging import Logger
from math import ceil, floor
from os.path import exists
from typing import Union, List

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed import ReduceOp
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, get_cosine_schedule_with_warmup

from dataset.dataset import LatticeTextDataset, dict_to_device, DataProcessCfg, CollateFn
from models import get_model_config, get_tokenizer, LatticeEncoder, LatticeBart, model_init
from train_lattice_encoder import save_checkpoint, select_best_checkpoint, config_optimizer, wrap_distributed_parallel
from utils.args import *
from utils.metric import WER, BLEU
from utils.utils import set_seed, RuntimeMetric, use_DP, use_DDP, setup_dist_environment, \
    cleanup_dist_environment, get_free_port, get_logger, show_args


def parse_args_and_setup_hyper_param(_args: List[str] = None) -> Namespace:
    psr = argparse.ArgumentParser()
    add_args_for_model_construct(psr)
    add_args_for_lr(psr)
    add_args_for_data_path(psr)
    add_args_for_data_process(psr)
    add_args_for_train_procedure(psr)
    add_args_for_seq2seq_train(psr)
    add_args_for_seq2seq_generation(psr)
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

    trainset, devset, testsets = None, None, []

    if not (args.train_lat is None and args.train_lat_ark is None) and (args.train_text is not None):
        trainset = LatticeTextDataset(lattice_pattern=args.train_lat,
                                      lattice_ark_pattern=args.train_lat_ark,
                                      text_pattern=args.train_text,
                                      process_cfg=process_cfg)

    if not (args.eval_lat is None and args.eval_lat_ark is None) and (args.eval_text is not None):
        devset = LatticeTextDataset(lattice_pattern=args.eval_lat,
                                    lattice_ark_pattern=args.eval_lat_ark,
                                    text_pattern=args.eval_text,
                                    process_cfg=process_cfg_for_test)

    if len(args.test_lat) == 0 and len(args.test_lat_ark) == 0:
        if logger:
            logger.error("No test data found.")

    if len(args.test_lat) > 0 and len(args.test_lat_ark) > 0:
        raise ValueError(f"Only one of `--test_lat` and `--test_lat_ark` of can be set. "
                         f"Got: test_lat: {args.test_lat}, test_lat_ark: {args.test_lat_ark}")

    test_lat = [None] * len(args.test_lat_ark) if len(args.test_lat) == 0 else args.test_lat
    test_lat_ark = [None] * len(args.test_lat) if len(args.test_lat_ark) == 0 else args.test_lat_ark
    if len(args.test_text) == 1:
        text_patterns = [args.test_text[0]] * len(test_lat)
    else:
        assert len(args.test_text) == len(test_lat)
        text_patterns = args.test_text

    for i, (_lat, _lat_ark, text) in enumerate(zip(test_lat, test_lat_ark, text_patterns)):
        testset = LatticeTextDataset(lattice_pattern=_lat, lattice_ark_pattern=_lat_ark, text_pattern=text,
                                     process_cfg=process_cfg_for_test)
        testsets.append(testset)

    return trainset, devset, testsets


def get_model_and_tokenizer(args: Namespace, logger: Logger, to_device):
    ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric, logger=logger)
    if ckpt_dir is not None:
        model = LatticeBart.from_pretrained(ckpt_dir)
        tokenizer = BartTokenizer.from_pretrained(args.exp_dir)
        if logger:
            logger.info(f"Load checkpoint from '{ckpt_dir}'")
    else:
        if args.default_config is not None:
            config = get_model_config(args.default_config)
            model = LatticeBart(config)
            tokenizer = get_tokenizer()

        elif args.bart_path is not None:
            model = LatticeBart.from_bart(bart_path=args.bart_path, lat_enc_layers=args.lat_enc_layers)
            tokenizer = BartTokenizer.from_pretrained(args.bart_path)

        elif args.lat_bart_path is not None:
            model = LatticeBart.from_pretrained(args.lat_bart_path)
            tokenizer = BartTokenizer.from_pretrained(args.lat_bart_path)

        else:
            if logger:
                logger.error("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")
            raise ValueError("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")

        # DDP will guarantee that parameters on different devices are the same! So please feel free to init it
        if args.init_model:
            if logger:
                logger.info("[!] Initialize model parameters.")
            model_init(model, std=model.config.init_std, logger=logger,
                       except_modules=(model.get_encoder().embed_tokens,))

        if getattr(args, "lat_encoder", None) is not None and getattr(args, "lat_encoder", None).strip() != "":
            best_ckpt = select_best_checkpoint(args.lat_encoder, use_metric="loss", logger=logger)
            if best_ckpt is not None:
                model.get_encoder().load_state_dict(
                    torch.load(join(best_ckpt, LatticeEncoder.MODEL_PT), map_location=lambda st, loc: st)
                )
                if logger:
                    logger.info(f"Load pretrained LatticeEncoder from '{best_ckpt}'")

    model.to(to_device)
    return model, tokenizer


def evaluate(model: Union[LatticeBart, DP], tokenizer: BartTokenizer, dataloader: DataLoader,
             args: Namespace, logger: Logger, save_to: str = None) -> dict:
    """ Calculate BLEU, CER, WER """
    _state = model.training
    model.eval()
    device = model.device if isinstance(model, LatticeBart) else model.module.device

    refers, hypos = [], []
    keys = []

    for inputs in dataloader:  # type: dict
        inputs.pop("decoder_input_ids")
        inputs.pop("decoder_attention_mask")

        key = inputs.pop("key", None)
        if key is not None:
            keys.extend(key)

        labels = inputs.pop("labels")
        batch_refers = tokenizer.batch_decode(labels, skip_special_tokens=True)
        refers.extend(batch_refers)

        inputs = dict_to_device(inputs, device)
        if isinstance(model, DP):
            outputs = model.module.generate(**inputs, max_length=args.max_len, min_length=args.min_len,
                                            num_beams=args.num_beams)
        else:
            outputs = model.generate(**inputs, max_length=args.max_len, min_length=args.min_len,
                                     num_beams=args.num_beams)
        batch_hypos = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        hypos.extend(batch_hypos)

    word_refers = [s.split() for s in refers]
    word_hypos = [s.split() for s in hypos]
    bleu = BLEU(word_refers, word_hypos)
    wer = WER(word_refers, word_hypos, is_wer=True, logger=logger)

    if save_to is not None:
        with open(join(args.exp_dir, save_to), "w", encoding="utf8") as f:
            if len(keys) != 0:
                for key, hypo in zip(keys, word_hypos):
                    f.write(f"{key}\t{' '.join(hypo)}\n")
            else:
                for hypo in word_hypos:
                    f.write(f"{' '.join(hypo)}\n")

    char_refers = [list(s) for s in refers]
    char_hypos = [list(s) for s in hypos]
    cer = WER(char_refers, char_hypos, is_wer=False, logger=logger)

    if _state:
        model.train()

    if logger:
        logger.info(f"Eval | BLEU: {bleu:.3f}")
    else:
        print(f"Eval | BLEU: {bleu:.3f}")

    return {"bleu": bleu, "wer": wer, "cer": cer}


def model_forward(model: Union[LatticeBart, DP, DDP], inputs: dict) -> dict:
    if isinstance(model, (DP, DDP)):
        outputs = model.forward(**inputs, reduction="none")
        nll_loss = outputs["loss"].mean()
    else:
        outputs = model.forward(**inputs)
        nll_loss = outputs["loss"]
    return {"loss": nll_loss}


def training(model: Union[LatticeBart, DP, DDP], tokenizer: BartTokenizer, train_loader: DataLoader,
             dev_loader: DataLoader, optimizer, scheduler, args: Namespace, logger: Logger, epoch_trained=0,
             runtime_metric: RuntimeMetric = None, writer: SummaryWriter = None):
    is_DP, is_DDP = isinstance(model, DP), isinstance(model, DDP)
    device = model.module.device if (is_DP or is_DDP) else model.device
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

    for epoch in range(epoch_trained + 1, args.epoch + 1):
        model.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        max_steps = len(train_loader)

        for step, inputs in enumerate(train_loader, start=1):
            inputs = dict_to_device(inputs, device)
            losses = model_forward(model, inputs)

            loss = losses["loss"]
            loss.backward()
            if mtc:
                mtc.avg_loss.append(loss.item())

            global_step = len(train_loader) * (epoch - 1) + step
            # if writer is not None:
            #     writer.add_scalars(main_tag="loss", tag_scalar_dict=losses, global_step=global_step)

            if logger and step % args.log_every == 0:
                lrs = [f"{group['lr']:.2e}" for group in optimizer.param_groups]
                lrs = [lrs[0], lrs[-1]] if len(lrs) > 2 else lrs
                lrs = " ~ ".join(lrs)
                msg = f"Ep {epoch} / {args.epoch} | st {step} / {max_steps} | lr: {lrs} | " \
                      f"avg loss: {mtc.avg_loss.avg_value:.2e}"
                logger.info(msg)

            if global_step % args.grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Evaluate after an epoch
        if mtc:
            _model = model.module if (is_DP or is_DDP) else model

            metrics = evaluate(_model, tokenizer, dev_loader, args=args, logger=logger)
            if writer is not None:
                writer.add_scalars(main_tag="Metric", tag_scalar_dict=metrics, global_step=len(train_loader) * epoch)

            metric = metrics[args.use_metric]

            if mtc.metric_is_better_than_ever(metric) or args.always_save:
                save_checkpoint(epoch=epoch, model=_model, metric=metric, optimizer=optimizer,
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
          args: Namespace, logger: Logger = None):
    if logger is None and rank == 0:
        logger = get_logger(args, "LatticeBartTrain")

    if trainset is None or len(trainset) == 0:
        if logger:
            logger.error("Empty trainset")
        return

    if devset is None or len(devset) == 0:
        if logger:
            logger.error("Empty devset")
        return

    if rank != -1:
        setup_dist_environment(rank, world_size, master_port)

    set_seed(args.seed)

    _use_DP, _use_DDP = use_DP(args), use_DDP(args)

    if _use_DDP:
        device = f"cuda:{rank}"
        model, tokenizer = get_model_and_tokenizer(args, logger, to_device=device)
        batch_size = args.train_batch // world_size
        num_workers = args.num_workers // world_size
    else:
        device = args.device
        model, tokenizer = get_model_and_tokenizer(args, logger, to_device=device)
        batch_size = args.train_batch
        num_workers = args.num_workers

    train_collate_fn = CollateFn(tokenizer=tokenizer, decoder_input=args.decoder_input).collate_fn
    dev_collate_fn = CollateFn(tokenizer=tokenizer, decoder_input=args.decoder_input).collate_fn

    train_loader = DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False if _use_DDP else True,
        collate_fn=train_collate_fn, sampler=DistributedSampler(trainset, seed=args.seed) if _use_DDP else None
    )
    dev_loader = DataLoader(
        devset, batch_size=args.eval_batch, num_workers=args.num_workers, collate_fn=dev_collate_fn
    )

    optimizer = config_optimizer(model, args)
    total_steps = ceil(args.epoch * len(trainset) / (args.train_batch * args.grad_acc_steps))
    if logger:
        logger.info(f"Total step: {total_steps}, warm-up steps: {floor(total_steps * 0.01)}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=floor(total_steps * 0.01),
        num_training_steps=floor(total_steps * 1.2)  # Prevent zero lr
    )

    mtc = RuntimeMetric(None, window_size=10, patience=args.patience,
                        decrease=(args.use_metric in ("cer", "wer", "loss")))
    epoch_trained = 0

    ### Fast Forward ###
    ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric, logger=logger)
    if ckpt_dir is not None:
        epoch_trained = int(re.match(r".*epoch_(?P<n>\d+).*", ckpt_dir).group("n"))
        step_trained = ceil(epoch_trained * len(trainset) / (args.train_batch * args.grad_acc_steps))

        optimizer.load_state_dict(torch.load(join(ckpt_dir, "optim.pt"), map_location=device))
        # It seems that moving params to cuda manually can be effect?
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        for _ in range(step_trained):
            scheduler.step()

        for i in range(epoch_trained):
            for _ in train_loader:
                break
            if logger:
                logger.info(f"Pass epoch {i + 1}")

        mtc = RuntimeMetric.load(join(ckpt_dir, "runtime_metric.json"))

    if _use_DP or _use_DDP:
        model = wrap_distributed_parallel(rank=rank, model=model, args=args, logger=logger)

    try:
        training(model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader, args=args,
                 logger=logger, optimizer=optimizer, scheduler=scheduler, epoch_trained=epoch_trained,
                 runtime_metric=mtc)
    except KeyboardInterrupt:
        if logger:
            logger.info("Interrupt by user.")

    if logger:
        logger.info("======== Finish training ========")

    if rank != -1:
        cleanup_dist_environment()


def test(testsets: List[LatticeTextDataset], args: Namespace, logger: Logger):
    if len(testsets) == 0:
        logger.error("No test data found.")
        return

    set_seed(args.seed)

    exp_dir = select_best_checkpoint(args.exp_dir, args.use_metric, logger=logger)
    if exp_dir is None:
        logger.error("No checkpoints found.")
        return

    try:
        # Loading model
        model = LatticeBart.from_pretrained(exp_dir).to(args.device).eval()
        tokenizer = BartTokenizer.from_pretrained(args.exp_dir)

        if use_DP(args):
            model = wrap_distributed_parallel(rank=-1, model=model, args=args, logger=logger)

        # Testing
        logger.info("======== Start testing ========")

        test_collate_fn = CollateFn(tokenizer, decoder_input="sent", return_keys=True).collate_fn

        for i, testset in enumerate(testsets):
            test_loader = DataLoader(
                testset, batch_size=args.eval_batch, num_workers=args.num_workers, collate_fn=test_collate_fn
            )
            evaluate(model, tokenizer, dataloader=test_loader, args=args, logger=logger, save_to=f"hypo_{i}.txt")

    except FileNotFoundError:
        logger.error("FileNotFoundError!")

        config_file = join(exp_dir, "config.json")
        pt_file = join(exp_dir, LatticeBart.MODEL_PT)

        if not exists(config_file):
            logger.error(f"config_file: `{config_file}` not found")
        if not exists(pt_file):
            logger.error(f"pt_file: `{pt_file}` not found")


def main(_args: List[str] = None):
    args = parse_args_and_setup_hyper_param(_args)
    os.makedirs(args.exp_dir, exist_ok=True)

    logger = get_logger(args, "LatticeBartMain")
    show_args(args, logger)

    trainset, devset, testsets = prepare_datasets(args, logger)
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
            train(rank=-1, world_size=-1, master_port=-1, trainset=trainset, devset=devset, args=args, logger=logger)
    else:
        logger.info("Training phase is passed.")

    if test_phase:
        test(testsets=testsets, args=args, logger=logger)
    else:
        logger.info("Testing phase is passed.")

    return args.exp_dir


if __name__ == '__main__':
    main()
