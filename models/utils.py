import logging

import torch.nn as nn
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer

from .encoder import LatticeEncoderLayer


def layer_wise_traverse(model: nn.Module, stop_layers=(LatticeEncoderLayer, BartEncoderLayer, BartDecoderLayer)):
    """ Travel across layers, from encoder to decoder """
    layers = []

    for name, child in model.named_children():
        if isinstance(child, stop_layers):
            logging.debug(f"[+] {name}, {child.__class__.__name__}")
            layers.append(child)
            continue
        layers += layer_wise_traverse(child, stop_layers)

    if len(layers) == 0:
        logging.debug(f"[+] {model.__class__.__name__}")
        layers.append(model)

    return layers
