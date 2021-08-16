"""
Training pipeline of LatticeBart
"""
import argparse
import logging
import time
from inspect import signature
from math import ceil, floor
from os.path import exists
from typing import Union

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, BartTokenizer, get_cosine_schedule_with_warmup

from models.lattice_bart import LatticeBart
from train_lattice_encoder import save_checkpoint, select_best_checkpoint, cos_similarity_mtx, contrastive_loss
from utils.args import *
from utils.dataset import LatticeTextDataset, dict_to_device
from utils.lamb import Lamb
from utils.metric import WER, BLEU
from utils.utils import LimitedQueue, EarlyStopping, set_seed, set_logger, model_init


def parse_args_and_setup_hyper_param() -> Namespace:
    psr = argparse.ArgumentParser()
    add_args_for_model_construct(psr)
    add_args_for_lr(psr)
    add_args_for_data_path(psr)
    add_args_for_data_process(psr)
    add_args_for_train_procedure(psr)
    add_args_for_seq2seq_train(psr)
    add_args_for_seq2seq_generation(psr)
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


def config_optimizer(model: LatticeBart, args: Namespace):
    name = args.optim.lower()
    if name == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif name == "adam":
        optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    elif name == "lamb":
        optim = Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9,
                     adam=True)
    else:
        raise ValueError(f"Unsupported optimizer called {args.optim}")
    return optim


def evaluate(model: Union[LatticeBart, nn.DataParallel], tokenizer: BartTokenizer, dataloader: DataLoader,
             args: Namespace) -> dict:
    """ Calculate BLEU, CER, WER """
    _state = model.training
    model.eval()

    refers, hypos = [], []

    for inputs in dataloader:
        inputs.pop("decoder_input_ids")
        inputs.pop("decoder_attention_mask")

        labels = inputs.pop("labels")
        batch_refers = tokenizer.batch_decode(labels, skip_special_tokens=True)
        refers.extend(batch_refers)

        inputs = dict_to_device(inputs, args.device)
        if isinstance(model, nn.DataParallel):
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
    wer = WER(word_refers, word_hypos, is_wer=True)

    char_refers = [list(s) for s in refers]
    char_hypos = [list(s) for s in hypos]
    cer = WER(char_refers, char_hypos, is_wer=False)

    if _state:
        model.train()

    logging.info(f"Eval | BLEU: {bleu:.3f}")
    return {"bleu": bleu, "wer": wer, "cer": cer}


def model_forward(model: Union[LatticeBart, nn.DataParallel], inputs: dict, eos_id: int, args: Namespace) -> dict:
    if args.contrast_method == "dropout":
        eos_pos = torch.where(inputs["input_ids"].eq(eos_id))

        # Forward twice
        if isinstance(model, nn.DataParallel):
            enc_param = signature(model.module.get_encoder().forward).parameters
            enc_inputs = {k: v for k, v in inputs.items() if k in enc_param}

            last_state1 = model.module.get_encoder().forward(**enc_inputs, output_hidden_states=True)
            last_state1 = last_state1.last_hidden_state[eos_pos]

            outputs = model.forward(**inputs, reduction="none")
            nll_loss = outputs["loss"].mean()
        else:
            enc_param = signature(model.get_encoder().forward).parameters
            enc_inputs = {k: v for k, v in inputs.items() if k in enc_param}

            last_state1 = model.get_encoder().forward(**enc_inputs, output_hidden_states=True)
            last_state1 = last_state1.last_hidden_state[eos_pos]

            outputs = model.forward(**inputs)
            nll_loss = outputs["loss"]

        last_state2 = outputs.encoder_last_hidden_state[eos_pos]
        sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
        contrast_loss = contrastive_loss(sim_mtx, reduction="mean")

        return {"loss": contrast_loss + nll_loss, "contrast": contrast_loss, "nll": nll_loss}

    else:
        # Forward once
        if isinstance(model, nn.DataParallel):
            outputs = model.forward(**inputs, reduction="none")
            nll_loss = outputs["loss"].mean()
        else:
            outputs = model.forward(**inputs)
            nll_loss = outputs["loss"]

        if args.contrast_method == "mask":
            eos_position = torch.where(inputs["input_ids"].eq(eos_id))
            enc_last_hidden = outputs["encoder_last_hidden_state"][eos_position]
            last_state1 = enc_last_hidden[::2]
            last_state2 = enc_last_hidden[1::2]
            sim_mtx = cos_similarity_mtx(last_state1, last_state2)  # [B, B]
            contrast_loss = contrastive_loss(sim_mtx, reduction="mean")
            return {"loss": contrast_loss + nll_loss, "contrast": contrast_loss, "nll": nll_loss}
        else:
            return {"loss": nll_loss, "nll": nll_loss}


def train(model: LatticeBart, tokenizer: BartTokenizer, trainset: LatticeTextDataset, devset: LatticeTextDataset,
          args: Namespace):
    # Collate function
    train_collate_fn = trainset.partial_collate_fn(tokenizer, n_lat=(2 if args.contrast_method == "mask" else 1))
    dev_collate_fn = devset.partial_collate_fn(tokenizer)

    # Dataloader
    train_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers,
                              collate_fn=train_collate_fn)
    dev_loader = DataLoader(devset, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers,
                            collate_fn=dev_collate_fn)

    # Model
    model.to(args.device)

    # Optimizer & Scheduler
    optimizer = config_optimizer(model, args)
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
    early_stop = EarlyStopping(patience=args.patience, decrease=(args.use_metric in ("cer", "wer", 'loss')))

    # TODO: 实现断点续训
    logging.info("==== Start training ====")

    for epoch in range(1, args.epoch + 1):
        model.train()
        # trainset.window_disturb(args.train_batch * args.grad_acc_steps)
        # trainset.shuffle()    # TAG: 直接对trainset调整似乎会出问题

        max_steps = len(train_loader)

        for step, inputs in enumerate(train_loader, start=1):
            inputs = dict_to_device(inputs, args.device)

            # if isinstance(model, nn.DataParallel):
            #     outputs = model.forward(**inputs, reduction="none")
            #     nll_loss = outputs["loss"].mean()
            # else:
            #     outputs = model.forward(**inputs)
            #     nll_loss = outputs["loss"]

            losses = model_forward(model, inputs, tokenizer.eos_token_id, args)

            loss = losses["loss"]
            loss.backward()
            avg_loss.append(loss.item())

            global_step = len(train_loader) * (epoch - 1) + step
            writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict=losses,
                global_step=global_step
            )

            if step % args.log_every == 0:
                lrs = ",".join([f"{group['lr']:.2e}" for group in optimizer.param_groups])
                logging.info(f"Ep {epoch} / {args.epoch} | st {step} / {max_steps} | lr: {lrs} | "
                             f"avg loss: {avg_loss.avg_value:.2e}")

            if global_step % args.grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Evaluate after an epoch
        metrics = evaluate(model, tokenizer, dev_loader, args)
        writer.add_scalars(
            main_tag="Metric", tag_scalar_dict=metrics,
            global_step=len(train_loader) * epoch
        )
        metric = metrics[args.use_metric]
        if best_metrics is None or \
                (best_metrics < metric and args.use_metric in ("bleu",)) or \
                (best_metrics > metric and args.use_metric in ("cer", "wer", "loss")):
            best_metrics = metric

            if isinstance(model, nn.DataParallel):
                save_checkpoint(epoch, metric, model.module, optimizer, tokenizer, args)
            else:
                save_checkpoint(epoch, metric, model, optimizer, tokenizer, args)

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

    if (args.train_lats is None) or (args.train_text is None):
        trainset = None
    else:
        cache_train = join(args.cache_dir, "cache_train")
        trainset = LatticeTextDataset(lattice_pattern=args.train_lats, text_pattern=args.train_text,
                                      cache_prefix=cache_train, min_nodes=args.min_nodes,
                                      max_nodes=args.max_nodes, **data_process)

    if (args.eval_lats is None) or (args.eval_text is None):
        devset = None
    else:
        cache_dev = join(args.cache_dir, "cache_dev")
        devset = LatticeTextDataset(lattice_pattern=args.eval_lats, text_pattern=args.eval_text,
                                    cache_prefix=cache_dev, min_nodes=0, max_nodes=args.max_nodes,
                                    **data_process)

    ######## Train ########
    # If no train or eval data, the training process will be passed
    if trainset is None or len(trainset) == 0 or devset is None or len(devset) == 0:
        logging.info("No valid training data. Training phase is passed.")
    else:
        if args.bart_path is not None:
            model = LatticeBart.from_bart(args.bart_path, args.depth_clip)
            tokenizer = BartTokenizer.from_pretrained(args.bart_path)
        elif args.lat_bart_path is not None:
            model = LatticeBart.from_pretrained(args.lat_bart_path)
            tokenizer = BartTokenizer.from_pretrained(args.lat_bart_path)
        else:
            logging.error("One of --bart_path and --lat_bart_path should be set")
            raise ValueError("One of --bart_path and --lat_bart_path should be set")

        if args.init_model:
            logging.info("[!] Initialize model parameters. Except Embedding")
            model_init(model, std=model.config.init_std, except_modules=(model.get_encoder().embed_tokens,))

        if getattr(args, "lat_encoder_pt", "") != "":
            model.get_encoder().load_state_dict(torch.load(args.lat_encoder_pt, map_location=lambda st, loc: st))
            logging.info(f"Load pretrained LatticeEncoder from '{args.lat_encoder_pt}'")

        # If you have multiple GPUs, why not use them?
        if "cuda" in args.device and torch.cuda.device_count() > 1:
            logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed training")
            model = nn.DataParallel(model)

        try:
            train(model, tokenizer, trainset, devset, args)
        except KeyboardInterrupt:
            logging.info("Interrupt by user.")

    ######## Test ########
    exp_dir = select_best_checkpoint(args)
    if exp_dir is None:
        logging.warning("No checkpoints found.")
    else:
        try:
            model = LatticeBart.from_pretrained(exp_dir).to(args.device).eval()
            tokenizer = BartTokenizer.from_pretrained(args.exp_dir)

            # If you have multiple GPUs, why not use them?
            if "cuda" in args.device and torch.cuda.device_count() > 1:
                logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed training")
                model = nn.DataParallel(model)

            # if (args.test_lats is None) or (args.test_text is None):
            #     logging.warning("No test data found.")
            #     return

            cache_test = join(args.cache_dir, "cache_test")
            testset = LatticeTextDataset(lattice_pattern=args.test_lats, text_pattern=args.test_text,
                                         cache_prefix=cache_test, min_nodes=0, max_nodes=args.max_nodes,
                                         **data_process)
            collate_fn = testset.partial_collate_fn(tokenizer)
            test_loader = DataLoader(testset, batch_size=args.eval_batch, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn)

            logging.info("==== Start testing ====")
            evaluate(model, tokenizer, test_loader, args)

        except FileNotFoundError:
            logging.error("FileNotFoundError!")

            config_file = join(exp_dir, "config.json")
            pt_file = join(exp_dir, LatticeBart.MODEL_PT)

            if not exists(config_file):
                logging.error(f"config_file: `{config_file}` not found")
            if not exists(pt_file):
                logging.error(f"pt_file: `{pt_file}` not found")


if __name__ == '__main__':
    main()
