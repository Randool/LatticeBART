"""
Train the encoder of LatticeBart in a contrastive-learning fashion
"""
import argparse
import glob
import logging
import time
from math import ceil, floor
from os.path import exists
from typing import Union

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, AdamW, get_cosine_schedule_with_warmup

from models.lattice_bart import LatticeBart, EncoderWrapper
from utils.args import *
from utils.dataset import LatticeTextDataset, dict_to_device
from utils.lamb import Lamb
from utils.utils import set_seed, set_logger, LimitedQueue, EarlyStopping, model_init


def parse_args_and_setup_hyper_param() -> Namespace:
    psr = argparse.ArgumentParser()
    add_args_for_model_construct(psr)
    add_args_for_lr(psr)
    add_args_for_data_path(psr)
    add_args_for_data_process(psr)
    add_args_for_train_procedure(psr)
    add_args_for_contrastive_learning(psr)
    args = psr.parse_args()

    for key, val in list(vars(args).items()):
        if val is None:
            continue
        if isinstance(val, str):
            setattr(args, key, val.strip())

    config_exp_dir(args)
    config_cache_dir(args)

    set_seed(args.seed)
    set_logger(args)

    return args


def config_optimizer(encoder: EncoderWrapper, args: Namespace):
    # TODO: 更好地控制每一层的学习率
    name = args.optim.lower()
    if name == "sgd":
        optim = torch.optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif name == "adam":
        optim = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    elif name == "lamb":
        optim = Lamb(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9,
                     adam=True)
    else:
        logging.error(f"Unsupported optimizer called {args.optim}")
        raise ValueError(f"Unsupported optimizer called {args.optim}")
    return optim


def save_checkpoint(epoch: int, metric, model, optimizer, tokenizer: BartTokenizer, args: Namespace):
    """
    Checkpoint contains:
        + model states
        + optimizer states
        + other states
    """
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
    # Optimizer state
    torch.save(optimizer.state_dict(), join(exp_dir, "optim.pt"))
    # Tokenizer
    tokenizer.save_pretrained(args.exp_dir)
    logging.info(f"Save checkpoints to {exp_dir}")


def select_best_checkpoint(args: Namespace):
    metrics = glob.glob(join(args.exp_dir, "epoch_*", "metric_*"))
    if len(metrics) == 0:
        return None

    best_exp_dir, best_metric = None, None

    for m in metrics:
        *_prefix, metric = m.split("/")
        metric = float(metric.split("_")[-1])

        if best_metric is None:
            best_metric = metric
            best_exp_dir = "/".join(_prefix)
        else:
            if args.use_metric in ("cer", "wer", "loss"):
                if best_metric > metric:
                    best_metric = metric
                    best_exp_dir = "/".join(_prefix)
            else:
                if best_metric < metric:
                    best_metric = metric
                    best_exp_dir = "/".join(_prefix)

    logging.info(f"Select checkpoint {best_exp_dir}")

    return best_exp_dir


def cos_similarity_mtx(vectors1, vectors2):
    """ vectors[1|2] of shape [B, D], return [B, B] """
    n1, n2 = vectors1.norm(dim=1, keepdim=True), vectors2.norm(dim=1, keepdim=True)
    eps = torch.finfo(n1.dtype).eps
    norm_1 = vectors1 / torch.clamp(n1, min=eps)
    norm_2 = vectors2 / torch.clamp(n2, min=eps)
    sim_mtx = torch.mm(norm_1, norm_2.transpose(0, 1))
    return sim_mtx


def contrastive_loss(sim_mtx, reduction="mean", improved=True):
    assert sim_mtx.shape[-2] == sim_mtx.shape[-1]

    if improved:
        """
        Contrastive loss improved by `Simpler, Faster, Stronger:
            Breaking The log-K Curse On Contrastive Learners With FlatNCE`
        """
        exp_mtx = torch.exp(sim_mtx)
        diag_val = torch.diagonal(exp_mtx)
        loss = torch.log(exp_mtx.sum(dim=1) - diag_val) - diag_val
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        index = torch.arange(0, sim_mtx.shape[-1], dtype=torch.long, device=sim_mtx.device)
        loss = F.cross_entropy(sim_mtx, index, reduction=reduction)

    return loss


def evaluate(encoder: EncoderWrapper, tokenizer: BartTokenizer, dataloader: DataLoader, args: Namespace) -> dict:
    _state = encoder.training
    encoder.eval()
    total_loss = 0

    for inputs in dataloader:
        inputs = dict_to_device(inputs, args.device)
        with torch.no_grad():
            if args.contrast_method == "dropout":
                last_state1 = encoder.forward(**inputs, eos_id=tokenizer.eos_token_id)
                last_state2 = encoder.forward(**inputs, eos_id=tokenizer.eos_token_id)
            else:
                last_state = encoder.forward(**inputs, eos_id=tokenizer.eos_token_id)
                last_state1 = last_state[::2]
                last_state2 = last_state[1::2]

        sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
        total_loss += contrastive_loss(sim_mtx).item()

    if _state:
        encoder.train()

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Eval | loss: {avg_loss:.2e}")
    return {"loss": avg_loss}


def train(encoder: Union[EncoderWrapper, nn.DataParallel], tokenizer: BartTokenizer, trainset: LatticeTextDataset,
          devset: LatticeTextDataset, args: Namespace):
    # Dataloader
    collate_fn = trainset.partial_collate_fn(tokenizer, n_lat=(2 if args.contrast_method == "mask" else 1))
    train_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, drop_last=isinstance(encoder, nn.DataParallel))
    dev_loader = DataLoader(devset, batch_size=args.eval_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # Model
    encoder.to(args.device)

    # Optimizer & Scheduler
    optimizer = config_optimizer(encoder, args)
    total_steps = ceil(args.epoch * len(trainset) / (args.train_batch * args.grad_acc_steps))
    logging.info(f"Total step: {total_steps}, warm-up steps: {floor(total_steps * 0.05)}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=floor(total_steps * 0.01),
        num_training_steps=total_steps
    )

    # Metrics
    best_metrics = None
    writer = SummaryWriter(log_dir=join(args.exp_dir, f"tensorboard_{time.strftime('%Y-%m-%d_%H-%M')}"))
    avg_loss = LimitedQueue(max_len=10)
    early_stop = EarlyStopping(patience=args.patience, decrease=(args.use_metric in ("cer", "wer", "loss")))

    logging.info("==== Start training ====")

    for epoch in range(1, args.epoch + 1):
        encoder.train()
        # trainset.window_disturb(args.train_batch * args.grad_acc_steps)
        # trainset.shuffle()     # TAG: 直接对trainset调整似乎会出问题

        max_steps = len(train_loader)
        tic = time.time()

        for step, inputs in enumerate(train_loader, start=1):
            inputs = dict_to_device(inputs, args.device)

            if args.contrast_method == "dropout":
                last_state1 = encoder.forward(**inputs, eos_id=tokenizer.eos_token_id)
                last_state2 = encoder.forward(**inputs, eos_id=tokenizer.eos_token_id)
            else:   # mask
                last_state = encoder.forward(**inputs, eos_id=tokenizer.eos_token_id)
                last_state1 = last_state[::2]
                last_state2 = last_state[1::2]

            sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
            loss = contrastive_loss(sim_mtx, reduction="mean")
            loss.backward()
            avg_loss.append(loss.item())

            global_step = len(train_loader) * (epoch - 1) + step
            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step)

            if step % args.log_every == 0:
                lrs = ",".join([f"{group['lr']:.2e}" for group in optimizer.param_groups])
                logging.info(f"Ep {epoch} / {args.epoch} | st {step} / {max_steps} | lr: {lrs} | "
                             f"avg loss: {avg_loss.avg_value:.2e}")

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

        span = (time.time() - tic) / 60
        logging.info(f"Epoch {epoch} training finish. Cost {span:.1f} min")

        # Evaluate
        metrics = evaluate(encoder, tokenizer, dev_loader, args)
        writer.add_scalars(
            main_tag="Metric", tag_scalar_dict=metrics,
            global_step=len(train_loader) * epoch
        )
        metric = metrics[args.use_metric]

        if best_metrics is None or \
                (best_metrics < metric and args.use_metric in ("bleu",)) or \
                (best_metrics > metric and args.use_metric in ("cer", "wer", "loss")):
            best_metrics = metric

            if isinstance(encoder, nn.DataParallel):
                save_checkpoint(epoch, metric, encoder.module, optimizer, tokenizer, args)
            else:
                save_checkpoint(epoch, metric, encoder, optimizer, tokenizer, args)

        if early_stop.stop(metric):
            logging.info(f"Early stopping after {epoch} epoch.")
            break


def main():
    args = parse_args_and_setup_hyper_param()

    ######## Dataset ########
    data_process = {
        "lower_case": True,
        "sort_by_len": args.sort_by_len == 1,
        "lm_scale": args.lm_scale,
        "ac_scale": args.ac_scale,
        "use_marginal": args.use_marginal == 1,
        "use_min_depth": args.use_min_depth == 1,
        "use_max_depth": args.use_max_depth == 1,
        "use_forward_backward": args.use_forward_backward == 1,
        "use_lattice_mask": args.use_lattice_mask == 1,
        "binary_mask": args.use_binary_mask == 1
    }

    if args.train_lats is None:
        trainset = None
    else:
        trainset = LatticeTextDataset(lattice_pattern=args.train_lats, text_pattern=args.train_text,
                                      cache_prefix=join(args.cache_dir, "cache_train"), min_nodes=args.min_nodes,
                                      max_nodes=args.max_nodes, **data_process)

    if args.eval_lats is None:
        devset = None
    else:
        devset = LatticeTextDataset(lattice_pattern=args.eval_lats, text_pattern=args.eval_text,
                                    cache_prefix=join(args.cache_dir, "cache_dev"), min_nodes=0,
                                    max_nodes=args.max_nodes, **data_process)

    ######## Train ########
    # If no train or eval data, the training process will be passed
    if trainset is None or len(trainset) == 0 or devset is None or len(devset) == 0:
        logging.info("No valid training data. Training phase is passed.")
    else:
        if args.bart_path is not None:
            encoder = LatticeBart.from_bart(args.bart_path, args.depth_clip).get_encoder()
            tokenizer = BartTokenizer.from_pretrained(args.bart_path)
        elif args.lat_bart_path is not None:
            encoder = LatticeBart.from_pretrained(args.lat_bart_path).get_encoder()
            tokenizer = BartTokenizer.from_pretrained(args.lat_bart_path)
        else:
            logging.error("One of --bart_path and --lat_bart_path should be set")
            raise ValueError("One of --bart_path and --lat_bart_path should be set")

        encoder = EncoderWrapper(encoder)

        if args.init_model:
            logging.info("[!] Initialize model parameters. Except Embedding")
            model_init(encoder, std=encoder.config.init_std, except_modules=())

        # If you have multiple GPUs, why not use them?
        if "cuda" in args.device and torch.cuda.device_count() > 1:
            logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed training")
            encoder = nn.DataParallel(encoder)

        try:
            train(encoder, tokenizer, trainset, devset, args)
        except KeyboardInterrupt:
            logging.info("Interrupt by user.")

    ######## Test ########
    # Gather metrics
    exp_dir = select_best_checkpoint(args)
    if exp_dir is None:
        logging.warning("No checkpoints found.")
    else:
        try:
            encoder = EncoderWrapper.from_pretrained(exp_dir).to(args.device).eval()
            tokenizer = BartTokenizer.from_pretrained(args.exp_dir)

            if "cuda" in args.device and torch.cuda.device_count() > 1:
                logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed testing")
                encoder = nn.DataParallel(encoder)

            # if args.test_lats is None:
            #     logging.warning("No test data found.")
            #     return

            cache_test = join(args.cache_dir, "cache_test")
            testset = LatticeTextDataset(lattice_pattern=args.test_lats, text_pattern=args.test_text,
                                         cache_prefix=cache_test, min_nodes=0, max_nodes=args.max_nodes,
                                         **data_process)
            collate_fn = testset.partial_collate_fn(tokenizer, n_lat=(2 if args.contrast_method == "mask" else 1))
            test_loader = DataLoader(testset, batch_size=args.eval_batch, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn)

            logging.info("==== Start testing ====")
            evaluate(encoder, tokenizer, test_loader, args)

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
