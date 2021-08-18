import os
import re
from argparse import ArgumentParser, Namespace
from os.path import join

import torch


def add_args_for_model_construct(parser: ArgumentParser):
    p = parser.add_argument_group("Model construct")
    p.add_argument("--default_config", type=str, choices=("tiny", "base", "large"), help="Select configurations")

    p.add_argument("--depth_clip", type=int, default=200, help="Lattice depth clip")
    p.add_argument("--lat_enc_layers", type=int, default=1, help="LatticeEncoder layers")

    p.add_argument("--bart_path", type=str, help="Pretrained Bart path")
    p.add_argument("--init_model", type=int, choices=(0, 1), default=0, help="Init all model's parameters")

    p.add_argument("--lat_bart_path", type=str, help="Pretrained LatticeBart path")
    p.add_argument("--lat_encoder", type=str, help="Pretrained LatticeEncoder path")

    p.add_argument("--checkpoint", type=str, help="checkpoint, epoch_n")


def add_args_for_lr(parser: ArgumentParser):
    p = parser.add_argument_group("Lr params")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate for pretrained parts")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--lr_decay", type=float, default=1.0, help="Layer-wise lr decay")
    p.add_argument("--grad_acc_steps", type=int, default=1, help="Steps to accumulate grad")
    p.add_argument("--clip", type=float, default=1, help="Gradient clip")


def add_args_for_data_path(parser: ArgumentParser):
    p = parser.add_argument_group("Dataset path")
    p.add_argument("--cache_dir", type=str, help="Path to cache processed data")

    p.add_argument("--train_lats", type=str, help="/path/to/lattice")
    p.add_argument("--eval_lats", type=str, help="/path/to/lattice")
    p.add_argument("--test_lats", type=str, nargs="*", default=[], help="/path/to/lattice")

    p.add_argument("--train_lat_ark", type=str, help="/path/to/lattice ark file")
    p.add_argument("--eval_lat_ark", type=str, help="/path/to/lattice ark file")
    p.add_argument("--test_lat_ark", type=str, nargs="*", default=[], help="/path/to/lattice ark file")

    p.add_argument("--train_text", type=str, help="/path/to/text")
    p.add_argument("--eval_text", type=str, help="/path/to/text")
    p.add_argument("--test_text", type=str, help="/path/to/text")


def add_args_for_data_process(parser: ArgumentParser):
    p = parser.add_argument_group("Data processing")
    p.add_argument("--sort_by_len", type=int, choices=(0, 1), default=1)

    p.add_argument("--lm_scale", type=float, default=1.0)
    p.add_argument("--ac_scale", type=float, default=1.0)

    p.add_argument("--min_nodes", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=256)

    p.add_argument("--use_marginal", type=int, choices=(0, 1), default=1)
    p.add_argument("--use_min_depth", type=int, choices=(0, 1), default=1)
    p.add_argument("--use_max_depth", type=int, choices=(0, 1), default=1)
    p.add_argument("--use_forward_backward", type=int, choices=(0, 1), default=1)
    p.add_argument("--use_lattice_mask", type=int, choices=(0, 1), default=1)
    p.add_argument("--use_binary_mask", type=int, choices=(0, 1), default=1)


def add_args_for_train_procedure(parser: ArgumentParser):
    p = parser.add_argument_group("Train-Eval-Test Procedure")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--optim", type=str, choices=["sgd", "adam", "lamb"], default="adam")
    p.add_argument("--exp_dir", type=str, required=True)

    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--train_batch", type=int, default=8)
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_every_step", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=5, help="Early stop")
    p.add_argument("--use_metric", type=str, choices=["bleu", "wer", "cer", "loss"], default="bleu")


CHOICE = ("dropout", "mask",)


def add_args_for_contrastive_learning(parser: ArgumentParser):
    p = parser.add_argument_group("Contrastive learning args")
    p.add_argument("--contrast_method", type=str, choices=CHOICE, default="dropout")
    # p.add_argument("--mask_parts", type=str, nargs="*", choices=LAT_INFO._fields)
    # p.add_argument("--mask_prob", type=float, default=0.1)


def add_args_for_seq2seq_train(parser: ArgumentParser):
    p = parser.add_argument_group("Seq2Seq training args")
    p.add_argument("--contrast_method", type=str, choices=CHOICE + ("none",), default="none")
    # p.add_argument("--n_lat", type=int, choices=(1, 2), default=1, help="Used in `collate_fn`.")


def add_args_for_seq2seq_generation(parser: ArgumentParser):
    p = parser.add_argument_group("Seq2Seq evaluation args")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--min_len", type=int, required=True)
    p.add_argument("--max_len", type=int, required=True)


####### CONFIG #######

def config_exp_dir(args: Namespace):
    # Normal configs
    parts = []
    if getattr(args, "bart_path", None) is not None:
        parts.append(f'bart_path-{getattr(args, "bart_path").rstrip("/").split("/")[-1]}')
    elif getattr(args, "lat_bart_path", None) is not None:
        value = re.sub("^.*bart_path-", "", getattr(args, "lat_bart_path")).split(",")[0]
        parts.append(f'bart_path-{value}')
    else:
        raise ValueError("One of --bart_path and --lat_bart_path should be set")

    if getattr(args, "init_model"):
        parts.append("init_model")

    for name in ("lr", "optim", "seed", "train_batch", "contrast_method"):
        if getattr(args, name, None) is not None:
            parts.append(f"{name}-{getattr(args, name)}")

    if getattr(args, "lat_encoder", "") != "":
        method = re.findall("(" + "|".join(CHOICE) + ")", getattr(args, "lat_encoder"))
        if len(method) == 1:
            method = method[0]
        elif len(method) == 0:
            raise ValueError(f"Can not find any choices ({CHOICE}) in `lat_encoder`")
        else:
            raise ValueError(f"Multiple choices on `{CHOICE}` in `lat_encoder`")
        parts.append(f"method-{method}")

    if "cuda" in args.device and torch.cuda.device_count() > 1:
        parts.append(f"gpu-{torch.cuda.device_count()}")

    # Data processing configs with particularity
    names = ("sort_by_len", "use_marginal", "use_min_depth", "use_max_depth",
             "use_forward_backward", "use_lattice_mask", "use_binary_mask")
    for name in names:
        if getattr(args, name) == 0:
            parts.append(f"{name}-{0}")

    args.exp_dir = join(args.exp_dir, ",".join(parts))

    os.makedirs(args.exp_dir, exist_ok=True)
    print(f"exp_dir: `{args.exp_dir}`")


def config_cache_dir(args: Namespace):
    if args.cache_dir is None:
        return
    parts = []
    for name in ('sort_by_len', 'lm_scale', 'ac_scale', 'min_nodes', 'max_nodes'):
        parts.append(f"{name}-{getattr(args, name)}")

    for name in (
            'use_marginal', 'use_min_depth', 'use_max_depth', 'use_forward_backward',
            'use_lattice_mask', 'use_binary_mask'
    ):
        if getattr(args, name) == 0:
            parts.append(f"{name}-{0}")

    args.cache_dir = join(args.cache_dir, ",".join(parts))

    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"cache_dir `{args.cache_dir}`")
