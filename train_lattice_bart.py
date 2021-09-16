"""
Training pipeline of LatticeBart
"""
import argparse
import logging
import time
from inspect import signature
from math import ceil, floor
from os.path import exists
from typing import Union, List

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, get_cosine_schedule_with_warmup

from dataset.dataset import LatticeTextDataset, dict_to_device, DataProcessCfg
from models.config import get_model_config, get_tokenizer
from models.encoder import LatticeEncoder
from models.lattice_bart import LatticeBart
from models.utils import model_init
from train_lattice_encoder import save_checkpoint, select_best_checkpoint, cos_similarity_mtx, contrastive_loss, \
    config_optimizer
from utils.args import *
from utils.metric import WER, BLEU
from utils.utils import set_seed, set_logger, RuntimeMetric


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

    set_seed(args.seed)
    set_logger(args)

    return args


def get_model_and_tokenizer(args: Namespace):
    ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric)
    if ckpt_dir is not None:
        model = LatticeBart.from_pretrained(ckpt_dir)
        tokenizer = BartTokenizer.from_pretrained(args.exp_dir)
        logging.info(f"Load checkpoint from '{ckpt_dir}'")
    else:
        if args.default_config is not None:
            model = LatticeBart(get_model_config(args.default_config))
            tokenizer = get_tokenizer()
        elif args.bart_path is not None:
            model = LatticeBart.from_bart(bart_path=args.bart_path, depth_clip=args.depth_clip,
                                          lat_enc_layers=args.lat_enc_layers)
            tokenizer = BartTokenizer.from_pretrained(args.bart_path)
        elif args.lat_bart_path is not None:
            model = LatticeBart.from_pretrained(args.lat_bart_path)
            tokenizer = BartTokenizer.from_pretrained(args.lat_bart_path)
        else:
            logging.error("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")
            raise ValueError("One of `--default_config`, `--bart_path` and `--lat_bart_path` should be set")

        if args.init_model:
            logging.info("[!] Initialize model parameters.")
            model_init(model, std=model.config.init_std, except_modules=(model.get_encoder().embed_tokens,))

        if getattr(args, "lat_encoder", None) is not None and getattr(args, "lat_encoder", None).strip() != "":
            best_ckpt = select_best_checkpoint(args.lat_encoder, "loss")
            model.get_encoder().load_state_dict(
                torch.load(join(best_ckpt, LatticeEncoder.MODEL_PT), map_location=lambda st, loc: st)
            )
            logging.info(f"Load pretrained LatticeEncoder from '{best_ckpt}'")

    # If you have multiple GPUs, why not use them?
    if "cuda" in args.device and torch.cuda.device_count() > 1:
        logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed training")
        model = nn.DataParallel(model)

    return model, tokenizer


def evaluate(model: Union[LatticeBart, nn.DataParallel], tokenizer: BartTokenizer, dataloader: DataLoader,
             args: Namespace, save_to: str = None) -> dict:
    """ Calculate BLEU, CER, WER """
    _state = model.training
    model.eval()

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


def train(model: LatticeBart, tokenizer: BartTokenizer, train_loader: DataLoader, dev_loader: DataLoader,
          optimizer, scheduler, writer: SummaryWriter, args: Namespace, epoch_trained=1,
          runtime_metric: RuntimeMetric = None):
    # Model
    model.to(args.device)

    # Metrics
    mtc = RuntimeMetric(None, window_size=10, patience=args.patience,
                        decrease=(args.use_metric in ("cer", "wer", 'loss'))) if runtime_metric is None \
        else runtime_metric

    logging.info("==== Start training ====")
    for epoch in range(epoch_trained, args.epoch + 1):
        model.train()

        max_steps = len(train_loader)

        for step, inputs in enumerate(train_loader, start=1):
            inputs = dict_to_device(inputs, args.device)
            losses = model_forward(model, inputs, tokenizer.eos_token_id, args)

            loss = losses["loss"]
            loss.backward()
            mtc.avg_loss.append(loss.item())

            global_step = len(train_loader) * (epoch - 1) + step
            writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict=losses,
                global_step=global_step
            )

            if step % args.log_every == 0:
                lrs = [f"{group['lr']:.2e}" for group in optimizer.param_groups]
                lrs = [lrs[0], lrs[-1]] if len(lrs) > 2 else lrs
                lrs = " ~ ".join(lrs)
                logging.info(f"Ep {epoch} / {args.epoch} | st {step} / {max_steps} | lr: {lrs} | "
                             f"avg loss: {mtc.avg_loss.avg_value:.2e}")

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
        if mtc.best_metrics is None or \
                (mtc.best_metrics <= metric and args.use_metric in ("bleu",)) or \
                (mtc.best_metrics >= metric and args.use_metric in ("cer", "wer", "loss")):
            mtc.best_metrics = metric

            _m = model.module if isinstance(model, nn.DataParallel) else model
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

    trainset = devset = None

    if not (args.train_lat is None and args.train_lat_ark is None) and args.train_text is not None:
        trainset = LatticeTextDataset(lattice_pattern=args.train_lat,
                                      lattice_ark_pattern=args.train_lat_ark,
                                      text_pattern=args.train_text,
                                      process_cfg=process_cfg)

    if not (args.eval_lat is None and args.eval_lat_ark is None) and args.eval_text is not None:
        devset = LatticeTextDataset(lattice_pattern=args.eval_lat,
                                    lattice_ark_pattern=args.eval_lat_ark,
                                    text_pattern=args.eval_text,
                                    process_cfg=process_cfg_for_test)

    #######################
    ######## Train ########
    #######################
    if trainset is None or len(trainset) == 0:
        logging.info("No valid training data. Training phase is passed.")
    elif devset is None or len(devset) == 0:
        logging.info("No valid validation data. Training phase is passed.")
    else:
        model, tokenizer = get_model_and_tokenizer(args)

        train_loader = DataLoader(
            trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers,
            collate_fn=trainset.partial_collate_fn(tokenizer, n_lat=(2 if args.contrast_method == "mask" else 1))
        )
        dev_loader = DataLoader(devset, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers,
                                collate_fn=devset.partial_collate_fn(tokenizer))

        optimizer = config_optimizer(model, args)
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
        writer = SummaryWriter(log_dir=join(args.exp_dir, f"tensorboard_{time.strftime('%Y-%m-%d_%H-%M')}"))

        ####################
        ### Fast Forward ###
        ####################
        ckpt_dir = select_best_checkpoint(args.exp_dir, use_metric=args.use_metric)
        if ckpt_dir is not None:
            epoch_trained = int(re.match(r".*epoch_(?P<n>\d+).*", ckpt_dir).group("n"))
            step_trained = ceil(epoch_trained * len(trainset) / (args.train_batch * args.grad_acc_steps))

            optimizer.load_state_dict(torch.load(join(ckpt_dir, "optim.pt"), map_location=args.device))
            # It seems that moving params to cuda manually can be effect?
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)

            for _ in range(step_trained):
                scheduler.step()

            for i in range(epoch_trained):
                for _ in train_loader:
                    break
                logging.info(f"Pass epoch {i + 1}")

            mtc = RuntimeMetric.load(join(ckpt_dir, "runtime_metric.json"))

        try:
            train(model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader, args=args,
                  writer=writer, optimizer=optimizer, scheduler=scheduler, epoch_trained=epoch_trained,
                  runtime_metric=mtc)
        except KeyboardInterrupt:
            logging.info("Interrupt by user.")

    ##############
    #### Test ####
    ##############
    exp_dir = select_best_checkpoint(args.exp_dir, args.use_metric)
    if exp_dir is None:
        logging.error("No checkpoints found.")
    else:
        if len(args.test_lat) == 0 and len(args.test_lat_ark) == 0:
            logging.error("No test data found.")
            return
        if len(args.test_lat) > 0 and len(args.test_lat_ark) > 0:
            logging.error(f"Only one `--test_lat` and `--test_lat_ark` of can be set. "
                          f"Got: test_lat: {args.test_lat}, test_lat_ark: {args.test_lat_ark}")
            return
        try:
            # Loading model
            model = LatticeBart.from_pretrained(exp_dir).to(args.device).eval()
            tokenizer = BartTokenizer.from_pretrained(args.exp_dir)

            if "cuda" in args.device and torch.cuda.device_count() > 1:
                logging.info(f"Found {torch.cuda.device_count()} GPUs, using distributed training")
                model = nn.DataParallel(model)

            # Testing
            logging.info("==== Start testing ====")

            test_lat = [None] * len(args.test_lat_ark) if len(args.test_lat) == 0 else args.test_lat
            test_lat_ark = [None] * len(args.test_lat) if len(args.test_lat_ark) == 0 else args.test_lat_ark

            for i, (_lat, _lat_ark) in enumerate(zip(test_lat, test_lat_ark)):
                testset = LatticeTextDataset(lattice_pattern=_lat, lattice_ark_pattern=_lat_ark,
                                             text_pattern=args.test_text, process_cfg=process_cfg_for_test)
                test_loader = DataLoader(
                    testset, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers,
                    collate_fn=testset.partial_collate_fn(tokenizer, n_lat=1, return_keys=True)
                )
                save_to = f"hypo_{i}.txt"
                evaluate(model=model, tokenizer=tokenizer, dataloader=test_loader, args=args, save_to=save_to)

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
