from collections import defaultdict
from logging import Logger
from typing import Dict, Union

import torch.nn as nn
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer

from utils.utils import log
from .encoder import LatticeEncoderLayer

_stop_layers = (LatticeEncoderLayer, BartEncoderLayer, BartDecoderLayer)


def layer_wise_traverse(model: nn.Module, stop_layers=_stop_layers):
    """ Travel across layers, from encoder to decoder. Recursion is stop when meeting layers exist in `stop_layers` """
    layers = []

    for name, child in model.named_children():
        if isinstance(child, stop_layers):
            layers.append(child)
            continue
        layers += layer_wise_traverse(child, stop_layers)

    if len(layers) == 0:
        layers.append(model)

    return layers


def config_layer_wise_lr(model: nn.Module, init_lr: float, lr_decay: float, stop_layers=_stop_layers,
                         logger: Logger = None) -> list:
    """ Specify per-layer learning rates """
    layers = layer_wise_traverse(model, stop_layers)

    # The parameters can be shared between layers, and we assign the maximum depth to parameters
    param_id_to_depth = {}  # type: Dict[int, int]

    for idx, layer in enumerate(reversed(layers)):
        for param in layer.parameters():
            param_id_to_depth[id(param)] = idx

    min_depth = min(param_id_to_depth.values())
    for pid, depth in param_id_to_depth.items():
        param_id_to_depth[pid] = depth - min_depth

    # Assign lr and archive parameters
    lr_to_param_list = defaultdict(list)

    for param in model.parameters():
        pid = id(param)
        if pid not in param_id_to_depth:
            log(f"Mismatch of param of shape {param.shape}", logger, "error")
            raise KeyError(f"Mismatch of param of shape {param.shape}")
        depth = param_id_to_depth[pid]
        lr = init_lr * (lr_decay ** depth)
        lr_to_param_list[lr].append(param)

    # Convert to a format suitable for torch.optim
    param_list_with_lr = []

    for lr, params in lr_to_param_list.items():
        param_list_with_lr.append({"params": params, "lr": lr})

    return param_list_with_lr


def model_init(model: nn.Module, std: float, except_modules: Union[list, tuple] = (), logger: Logger = None):
    # Init function
    def _init_weights(m):
        if m in except_modules:
            if logger:
                logger.debug(f"Module ({m}) gets rid of initialization")
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
