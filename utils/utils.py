import logging
import os
import random
from argparse import Namespace
from os.path import join
from typing import Union

import numpy as np
import torch
import torch.nn as nn


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


class LimitedQueue(list):
    def __init__(self, max_len: int = -1):
        super().__init__()
        self.max_len = max_len

    def append(self, item: float):
        if self.max_len != -1 and len(self) >= self.max_len:
            self.pop(0)
        super().append(item)

    @property
    def is_full(self):
        return len(self) == self.max_len

    @property
    def avg_value(self):
        if len(self) == 0:
            return 0
        return sum(self) / len(self)


class EarlyStopping:
    def __init__(self, patience: int, decrease=True):
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


def model_init(model: nn.Module, std: float, except_modules: Union[list, tuple] = ()):
    # Init function
    def _init_weights(m):
        if m in except_modules:
            logging.debug(f"Module ({m}) gets rid of initialization")
            return
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

    model.apply(_init_weights)
