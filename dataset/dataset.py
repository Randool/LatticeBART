import glob
import logging
import os
import pickle
import random
from collections import namedtuple
from os.path import exists, join, split
from typing import Union, Dict, Any, List

import networkx as nx
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import BartTokenizer, BatchEncoding

from dataset.parse import parse_lattice_file, line_graph_process, parse_lattice_from_address
from utils.tokenizer import word_based_tokenize, HomophoneReplacer

MAX_NODEs = int(1e3)

"""
Graph attrs:
    - node attrs: 'word', 'm' (marginal)
    - edge attrs: 'f' (forward), 'b' (backward)
"""


def dict_to_device(inputs: Union[dict, BatchEncoding], device=None) -> Union[dict, BatchEncoding]:
    """
    transformers的tokenizer的encode结果往往是一个dict，此方法将dict中所有的tensor传送到指定device上
    """
    for k in inputs:
        if isinstance(inputs[k], Tensor):
            inputs[k] = inputs[k].to(device)
    return inputs


def _load_cache(cache_file: str):
    with open(cache_file, "rb") as f:
        _data = pickle.load(f)
    return _data


def _cache_data(data, cache_file: str):
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


class _KeyBased:
    items: Dict[str, Any]
    n_files = 0

    def keys(self):
        yield from self.items.keys()

    def __getitem__(self, key):
        return self.items[key]

    def __len__(self):
        return len(self.items)


class _LatticeDataset(_KeyBased):
    """ Dataset to parse only kaldi's compact lattice files """

    def __init__(self, lattice_file_pattern: str, cache_dir: str = None, min_nodes: int = 0,
                 max_nodes: int = MAX_NODEs, lower_case=True, lm_scale=1.0, ac_scale=1.0):
        self.items = {}  # type: Dict[str, List[nx.DiGraph]]

        if cache_dir is None:
            cache_dir = None
        else:
            key = f"min_{min_nodes},max_{max_nodes},low_{lower_case},lm_{lm_scale:.2e},ac_{ac_scale:.2e}"
            cache_dir = join(cache_dir, key)
            os.makedirs(cache_dir, exist_ok=True)

        files = sorted(glob.glob(lattice_file_pattern))
        self.n_files = len(files)

        for file in files:
            if cache_dir is not None:
                filename = join(cache_dir, f"{split(file)[-1]}.cache")
                if exists(filename):
                    _lattices = _load_cache(filename)
                    logging.info(f"Load {len(_lattices)} lattices from {filename}.")
                else:
                    _lattices = self._load_and_filter(file, min_nodes, max_nodes, lower_case, lm_scale, ac_scale)
                    _cache_data(_lattices, filename)
                    logging.info(f"Cache {len(_lattices)} lattices to {filename}.")
            else:
                _lattices = self._load_and_filter(file, min_nodes, max_nodes, lower_case, lm_scale, ac_scale)

            for lat_id, lat in _lattices.items():
                # TODO: 确保lattice之间的差异性
                if lat_id in self.items:
                    self.items[lat_id].append(lat)
                else:
                    self.items[lat_id] = [lat]

        # Read data
        if len(self.items) == 0:
            raise RuntimeError("No lattice files found")

        logging.info(f"{self.__class__.__name__}: {len(self.items)} lattice ids.")

    @staticmethod
    def _load_and_filter(file: str, min_nodes: int, max_nodes: int, lower_case: bool, lm_scale: float, ac_scale: float):
        # Do data process
        _lattices = parse_lattice_file(file, to_line_graph=True, lower_case=lower_case,
                                       lm_scale=lm_scale, ac_scale=ac_scale)
        for lattice_id, lat in list(_lattices.items()):
            if len(lat) < min_nodes or len(lat) > max_nodes:
                del _lattices[lattice_id]
        logging.info(f"Load {len(_lattices)} lattices from {file}.")
        return _lattices


class _LatticeArkDataset(_KeyBased):
    def __init__(self, lattice_ark_pattern: str):
        self.items = {}

        files = sorted(glob.glob(lattice_ark_pattern))
        self.n_files = len(files)

        for file in files:
            with open(file, encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    key, *items = line.split()
                    self.items[key] = items

        logging.info(f"{self.__class__.__name__}: {len(self.items)} lattice ids.")


class __TextDataset(_KeyBased):
    """ Dataset to parse only text file """

    def __init__(self, text_file_pattern: str, lower_case=True, token_split=False):
        self.items = {}

        # Read data
        files = sorted(glob.glob(text_file_pattern))
        self.n_files = len(files)
        if len(files) == 0:
            raise FileNotFoundError("No text files found")

        for file in files:
            items = self._parse_text_file(file, lower_case=lower_case, token_split=token_split)
            for key, text in items:
                self.items[key] = text
            logging.info(f"Load {len(items)} texts from {file}.")

        logging.info(f"{self.__class__.__name__}: {len(self.items)} items.")

    @staticmethod
    def _parse_text_file(text_file: str, lower_case=True, token_split=True):
        """ format: text_id a b c ... """
        items = []
        empty_sent = 0
        with open(text_file, encoding="utf8") as f:
            for line in f:
                if len(line) <= 1 or line.isspace():
                    continue
                tokens = line.split(maxsplit=1)
                if len(tokens) == 1:
                    _id, tokens = tokens[0], ""
                    empty_sent += 1
                else:
                    _id, tokens = tokens

                if lower_case:
                    tokens = tokens.lower()
                if token_split:
                    tokens = tokens.split()
                items.append((_id, tokens))
        # logging.warning(f"{empty_sent} empty sentences found in {text_file}")
        return items


class _OracleDataset(__TextDataset):
    def __init__(self, text_file_pattern: str, lower_case=True):
        super().__init__(text_file_pattern, lower_case=lower_case, token_split=True)


class _TextDataset(__TextDataset):
    def __init__(self, text_file_pattern: str, lower_case=True):
        super().__init__(text_file_pattern, lower_case=lower_case, token_split=False)


DataProcessCfg = namedtuple("DataProcessCfg", (
    "min_nodes", "max_nodes", "lower_case", "ac_scale", "lm_scale", "use_marginal", "use_min_depth",
    "use_max_depth", "use_forward_backward", "use_lattice_mask", "binary_mask", "homophone_replace"
))

_default_cfg = DataProcessCfg(1, 200, True, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0)


class LatticeTextDataset(Dataset):
    def __init__(
            self, lattice_pattern: str = None, lattice_ark_pattern: str = None, text_pattern: str = None,
            cache_dir: str = None, process_cfg: DataProcessCfg = _default_cfg
    ):
        self.process_cfg = process_cfg

        if lattice_pattern is None and lattice_ark_pattern is None:
            raise ValueError("One of --lattice_pattern or --lattice_ark should be given")

        if lattice_pattern is not None:
            lattices = _LatticeDataset(lattice_pattern, cache_dir=cache_dir, min_nodes=self.process_cfg.min_nodes,
                                       max_nodes=self.process_cfg.max_nodes, lower_case=self.process_cfg.lower_case,
                                       ac_scale=self.process_cfg.ac_scale, lm_scale=self.process_cfg.lm_scale)
            self.n_files = lattices.n_files
        else:
            lattices = _LatticeArkDataset(lattice_ark_pattern)
            self.n_files = lattices.n_files

        if text_pattern is None:
            self.items = [[key, lattices[key], None] for key in lattices.keys()]
        else:
            transcripts = _TextDataset(text_pattern, lower_case=self.process_cfg.lower_case)
            common_keys = set(lattices.keys()) & set(transcripts.keys())
            self.items = [[key, lattices[key], transcripts[key]] for key in lattices.keys() if key in common_keys]

        logging.info(f"{self.__class__.__name__}: {len(self)} items.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def sort_by_length(self, increase=True):
        self.items = sorted(self.items, key=lambda x: max(map(len, x[1])), reverse=not increase)

    def shuffle(self):
        random.shuffle(self.items)

    def window_disturb(self, window_size: int):
        huff = window_size // 2
        for i in range(0, len(self), huff):
            j = i + window_size
            pairs = self.items[i:j]
            random.shuffle(pairs)
            self.items[i:j] = pairs

    def partial_collate_fn(self, tokenizer: BartTokenizer, n_lat=1, return_keys=False,
                           process_cfg: DataProcessCfg = None):
        if process_cfg is None:
            process_cfg = self.process_cfg

        homophone_replacer = HomophoneReplacer()

        def padding_seqs(seqs: list, pad, max_len, tensor_type):
            for i, seq in enumerate(seqs):
                seqs[i] = max_len * [pad] if (seq is None) else seq + (max_len - len(seq)) * [pad]
            return torch.tensor(seqs, dtype=tensor_type)

        def padding_mats(mats: list, pad, shape, tensor_type):
            _mats = torch.zeros(len(mats), *shape, dtype=tensor_type).fill_(pad)
            for i, mat in enumerate(mats):
                if mat is None:
                    continue
                m, n = mat.shape
                _mats[i, :m, :n] = mat
            return _mats

        def collate_fn(batch_data: list) -> dict:
            keys = []

            batch_input_seqs = []
            batch_min_depth = []
            batch_max_depth = []
            batch_lat_mask = []
            batch_fb = []
            batch_marginal = []

            batch_texts = []

            for key, lats, text in batch_data:
                if return_keys:
                    keys.append(key)

                idx = [i % len(lats) for i in range(max(len(lats), n_lat))]  # Padding lats if not enough
                random.shuffle(idx)
                idx = idx[:n_lat]
                batch_texts.extend([text] * n_lat)

                # Process nx.Digraph-formatted lattice
                for i in idx:
                    lat = lats[i]
                    if isinstance(lat, nx.DiGraph):
                        lats[i] = lat = line_graph_process(
                            lat, use_marginal=(random.random() <= process_cfg.use_marginal),
                            min_depth=(random.random() <= process_cfg.use_min_depth),
                            max_depth=(random.random() <= process_cfg.use_max_depth),
                            forward_backward=(random.random() <= process_cfg.use_forward_backward),
                            use_lattice_mask=(random.random() <= process_cfg.use_lattice_mask),
                            binary_mask=(random.random() <= process_cfg.binary_mask)
                        )
                    elif isinstance(lat, str):  # ark format: <file>:<bias>:<delta>
                        lattice_id, lat = parse_lattice_from_address(lat, to_line_graph=True,
                                                                     lower_case=process_cfg.lower_case,
                                                                     lm_scale=process_cfg.lm_scale,
                                                                     ac_scale=process_cfg.ac_scale)

                        # `key.startswith(lattice_id)` is used when do contrastive learning
                        assert lattice_id == key or key.startswith(lattice_id), \
                            f"Lattice id mismatch: lattice_id ({lattice_id}, key ({key}))"

                        lat = line_graph_process(
                            lat, use_marginal=(random.random() <= process_cfg.use_marginal),
                            min_depth=(random.random() <= process_cfg.use_min_depth),
                            max_depth=(random.random() <= process_cfg.use_max_depth),
                            forward_backward=(random.random() <= process_cfg.use_forward_backward),
                            use_lattice_mask=(random.random() <= process_cfg.use_lattice_mask),
                            binary_mask=(random.random() <= process_cfg.binary_mask)
                        )

                    input_text, marginal, min_depth, max_depth, fb, lattice_mask = lat
                    batch_input_seqs.append(input_text)
                    batch_min_depth.append(min_depth)
                    batch_max_depth.append(max_depth)
                    batch_lat_mask.append(lattice_mask)
                    batch_fb.append(fb)
                    batch_marginal.append(marginal)

            # Replace homophone
            if process_cfg.homophone_replace > 0:
                batch_input_seqs = [homophone_replacer.process_sent(s, ratio=process_cfg.homophone_replace) for s in
                                    batch_input_seqs]

            inputs = word_based_tokenize(batch_input_seqs, tokenizer, lower=True,
                                         return_attention=False, to_tensor=True)

            if return_keys:
                inputs["key"] = keys

            B, max_len = inputs["input_ids"].shape
            mat_shape = (max_len, max_len)

            inputs["min_depth"] = padding_seqs(batch_min_depth, MAX_NODEs, max_len, torch.long)
            inputs["max_depth"] = padding_seqs(batch_max_depth, MAX_NODEs, max_len, torch.long)
            inputs["lattice_mask"] = padding_mats(batch_lat_mask, 0, mat_shape, torch.long)
            inputs["marginals"] = padding_seqs(batch_marginal, 0, max_len, torch.float)
            inputs["forward_backward"] = padding_mats(batch_fb, 0, mat_shape, torch.float)
            inputs["batch_idx"] = torch.arange(B, dtype=torch.long)

            if batch_texts[0] is not None:
                outputs = tokenizer.batch_encode_plus(batch_texts, padding=True, return_tensors="pt")
                inputs["decoder_input_ids"] = outputs["input_ids"][:, :-1].contiguous()
                inputs["decoder_attention_mask"] = outputs["attention_mask"][:, :-1].contiguous()
                inputs["labels"] = outputs["input_ids"][:, 1:].contiguous()

            return inputs

        return collate_fn
