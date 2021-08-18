import json
import logging
import os
import random
from argparse import Namespace
from collections import namedtuple
from os.path import join
from typing import List

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_logger(args: Namespace):
    os.makedirs(args.exp_dir, exist_ok=True)
    logging.basicConfig(filename=join(args.exp_dir, "log.txt"), level=logging.INFO,
                        datefmt='%H:%M:%S', format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.info("=" * 30)
    for _arg in vars(args):
        logging.info(f"{_arg:20s}\t{getattr(args, _arg)}")
    logging.info("=" * 30)


class LimitedQueue:
    def __init__(self, max_len: int = -1, items: List[float] = None):
        self.max_len = max_len
        self.items: List[float] = [] if items is None else items

    def append(self, item: float):
        while self.max_len != -1 and len(self) >= self.max_len:
            self.pop(0)
        self.items.append(item)

    def pop(self, index=-1):
        return self.items.pop(index)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        yield from self.items

    def __str__(self):
        return str(self.items)

    def __repr__(self):
        return str(self.items)

    @property
    def is_full(self):
        return len(self) == self.max_len

    @property
    def avg_value(self):
        if len(self) == 0:
            return 0
        return sum(self.items) / len(self)


class EarlyStopping:
    def __init__(self, patience: int = 1, decrease=True):
        self.metrics = LimitedQueue(patience)
        self.decrease = decrease

    def stop(self, metric: float) -> bool:
        if not self.metrics.is_full:
            self.metrics.append(metric)
            return False

        if self.decrease:
            satisfy = any([metric < x for x in self.metrics])
        else:
            satisfy = any([metric > x for x in self.metrics])
        self.metrics.append(metric)

        return not satisfy

    def __repr__(self):
        return f"EarlyStopping<decrease={self.decrease},{repr(self.metrics)}>"


class RuntimeMetric:
    def __init__(self, best_metrics=None, window_size=10, patience=1, decrease=True):
        self.best_metrics = best_metrics
        self.avg_loss = LimitedQueue(window_size)
        self.early_stop = EarlyStopping(patience, decrease)

    def dump(self, filename: str):
        tmp = {
            "best_metrics": self.best_metrics,
            "window_size": self.avg_loss.max_len,
            "patience": self.early_stop.metrics.max_len,
            "decrease": self.early_stop.decrease,
            "avg_loss.items": self.avg_loss.items,
            "early_stop.metrics.items": self.early_stop.metrics.items
        }
        with open(filename, "w", encoding="utf8") as f:
            json.dump(tmp, f, indent=1)

    @classmethod
    def load(cls, filename: str):
        with open(filename, encoding="utf8") as f:
            tmp = json.load(f)
            avg_loss_items = tmp.pop("avg_loss.items")
            early_stop_metrics_items = tmp.pop("early_stop.metrics.items")
        ins = cls(**tmp)
        ins.avg_loss.items = avg_loss_items
        ins.early_stop.metrics.items = early_stop_metrics_items
        return ins


StatePack = namedtuple("StatePack", ("epoch", "runtime_metric", "optimizer", "scheduler", "train_loader", "writer"))
