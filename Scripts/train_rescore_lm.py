import argparse
import math
import os
from argparse import Namespace
from logging import Logger
from os.path import join

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, dict_to_device
from models.lstm_lm import Vocab, RNNModel
from utils.args import add_args_for_lr, add_args_for_train_procedure
from utils.utils import get_logger, show_args


def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    psr = parser.add_argument_group("Data")
    psr.add_argument("--train_text", type=str, required=True)
    psr.add_argument("--dev_text", type=str, required=True)

    psr = parser.add_argument_group("Model")
    psr.add_argument("--embed_size", type=int, default=128)
    psr.add_argument("--hidden_size", type=int, default=128)
    psr.add_argument("--n_layers", type=int, default=2)
    psr.add_argument("--dropout", type=float, default=0.2)

    add_args_for_lr(parser)
    add_args_for_train_procedure(parser)

    args = parser.parse_args(_args)
    return args


def prepare_datasets(args: Namespace, logger: Logger):
    vocab = Vocab(args.train_text)
    trainset = TextDataset(args.train_text)
    devset = TextDataset(args.dev_text)
    if logger is not None:
        logger.info(f"Train: len({len(trainset)})")
        logger.info(f"Dev: len({len(devset)})")
    return vocab, trainset, devset


def save_checkpoint(epoch: int, model, metric, vocab: Vocab, args: Namespace, logger: Logger):
    exp_dir = join(args.exp_dir, f"epoch_{epoch}")
    os.makedirs(exp_dir, exist_ok=True)

    model.save_pretrained(exp_dir)
    open(join(exp_dir, f"metric_{metric:.2e}"), "w")
    # Tokenizer
    vocab.save_pretrained(args.exp_dir)

    if logger:
        logger.info(f"Save checkpoints to {exp_dir}")


def main(_args=None):
    args = parse_args(_args)
    logger = get_logger(args, "LSTM")
    show_args(args, logger)
    vocab, trainset, devset = prepare_datasets(args, logger)
    collate_fn = vocab.collate_fn

    logger.info(f"Vocab size: {len(vocab)}")
    model = RNNModel(len(vocab), args.embed_size, args.hidden_size, args.n_layers, args.dropout).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(trainset, batch_size=args.train_batch, num_workers=2, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(devset, batch_size=args.eval_batch, num_workers=2, shuffle=False, collate_fn=collate_fn)

    for epoch in range(1, args.epoch + 1):
        model.train()
        steps = len(train_loader)
        for i, batch in enumerate(train_loader):
            batch.pop("length")
            batch = dict_to_device(batch, args.device)
            outputs = model.forward(**batch)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.log_every == 0:
                logger.info(f"Epoch: {epoch} / {args.epoch} | st: {i} / {steps} | Loss: {loss.item():.2e}")

        model.eval()
        avg_ppl = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = dict_to_device(batch, args.device)
                ppl = model.calculate_logprob(**batch)
                ppl = [math.exp(p) for p in ppl]
                avg_ppl += sum(ppl)
        avg_ppl /= len(devset)
        logger.info(f"Eval PPL: {avg_ppl:.2e}")
        save_checkpoint(epoch, model, metric=avg_ppl, vocab=vocab, args=args, logger=logger)


if __name__ == "__main__":
    main()
