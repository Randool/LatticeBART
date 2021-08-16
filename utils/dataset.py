import glob
import gzip
import logging
import pickle
import random
import sys
from collections import namedtuple
from itertools import count
from os.path import exists
from typing import Union, Tuple, Dict, Any, List

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import BartTokenizer, BatchEncoding

from utils.tokenizer import word_based_tokenize
from .symbols import *

_EPS = sys.float_info.min
MAX_NODEs = int(1e3)
WARNED = True

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


def lattice_to_line_graph(G: nx.DiGraph) -> nx.DiGraph:
    """ Convert node-labeled lattice to edge-labeled lattices """
    L = nx.line_graph(G)  # type: nx.DiGraph
    L.add_node((0, 0), word=BOS)

    for edge in G.edges:
        start, end = edge
        attrs = G[start][end]
        word = attrs.pop("word")
        nx.set_node_attributes(L, {edge: {"word": word}})
        if start == 0:
            L.add_edge((0, 0), edge, **attrs)
        else:
            for pred in L.predecessors(edge):
                nx.set_edge_attributes(L, {(pred, edge): attrs})

    # Replace eos node
    eos_edges = [edge for edge in L.edges if L.nodes[edge[1]]['word'] == EOS]
    eos_node = eos_edges[0][1]
    for edge in eos_edges:
        if edge[1] != eos_node:
            L.add_edge(edge[0], eos_node, f=_EPS)
            if edge in L.edges:
                L.remove_edge(*edge)
            if edge[1] in L:
                L.remove_node(edge[1])

    # Rename node_id
    rename = {j: i for i, j in enumerate(nx.topological_sort(L))}
    L = nx.relabel_nodes(L, rename, copy=False)
    return L


def scoring_line_graph(L: nx.DiGraph, topo_order: list = None) -> nx.DiGraph:
    """ Assign marginal/forward/backward score on the given line graph """
    if topo_order is None:
        topo_order = list(nx.topological_sort(L))
    nodes, edges = L.nodes, L.edges

    # Forward scores and marginal scores can be calculated simultaneously
    nx.set_node_attributes(L, _EPS, "m")
    nodes[topo_order[0]]['m'] = 1.0

    for i in topo_order:
        m_i = nodes[i]["m"]
        succ_edges = L[i]
        if len(succ_edges) == 0:
            continue
        """In `kaldi/src/lat/kaldi-lattice.cc`, the forward (alpha) score is calculated through:
            (*alpha)[0] = 0.0;
            for (StateId s = 0; s < num_states; s++) {
              double this_alpha = (*alpha)[s];
              for (ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
                const Arc &arc = aiter.Value();
                double arc_like = -(arc.weight.Weight().Value1() + arc.weight.Weight().Value2());
                (*alpha)[arc.nextstate] = LogAdd((*alpha)[arc.nextstate], this_alpha + arc_like);
              }
            }
        Therefore, the weight of kaldi's CompactLattice is negative logarithm.
        """
        fs = np.array([succ_edges[j]["f"] for j in succ_edges])
        fs = np.exp(fs - max(fs))
        fs = (fs / sum(fs)).tolist()

        for j, f in zip(succ_edges, fs):
            succ_edges[j]["f"] = f
            nodes[j]["m"] += m_i * f

    # Calculate backward scores
    for edge in edges:
        i, j = edge
        edges[edge]["b"] = edges[edge]["f"] * nodes[i]["m"] / nodes[j]["m"]

    return L


def min_max_depth(G: nx.DiGraph, topo_order: list = None) -> Tuple[dict, dict]:
    """ Return the longest distance of nodes from source """
    if topo_order is None:
        topo_order = list(nx.topological_sort(G))
    min_depths = {}
    max_depths = {}
    for node in topo_order:
        if len(G.pred[node]) == 0:
            max_depths[node] = min_depths[node] = 0
        else:
            min_depths[node] = min([min_depths[pred] for pred in G.pred[node]]) + 1
            max_depths[node] = max([max_depths[pred] for pred in G.pred[node]]) + 1
    return min_depths, max_depths


def mtx_relative_position(L: nx.DiGraph) -> torch.Tensor:
    assert all(isinstance(node, int) for node in L.nodes), "All node_id should be map to int"
    distance = nx.all_pairs_shortest_path_length(L)
    mtx = torch.zeros(len(L), len(L), dtype=torch.int).fill_(torch.iinfo(torch.int).min)
    for i, succs in distance:
        for j, d in succs.items():
            mtx[i, j] = d
            mtx[j, i] = -d
    return mtx


def binary_reachability_mask(L: nx.DiGraph) -> torch.LongTensor:
    relation = mtx_relative_position(L)
    return relation.ne(torch.iinfo(torch.int).min).long()


def probabilistic_reachability_mask(L: nx.DiGraph) -> torch.FloatTensor:
    raise NotImplementedError


def mtx_transfer(L: nx.DiGraph) -> torch.Tensor:
    mtx = torch.zeros(len(L), len(L), dtype=torch.float)
    for i in L:
        for j, attrs in L[i].items():
            mtx[i, j] = attrs["f"]
            mtx[j, i] = attrs["b"]
    return mtx


def parse_lattice_file(lattice_file: str, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) \
        -> Dict[str, nx.DiGraph]:
    """ Parse lattice file and return line-graphs """
    lattices = {}
    terminate_states = []

    reader = gzip.open if lattice_file.endswith(".gz") else open
    lattice_id = None

    with reader(lattice_file, "rt", encoding="utf8") as f:
        for line in f:
            items = line.strip().split()

            if len(items) == 1 and lattice_id is None:
                lattice_id = items[0]
                G = nx.DiGraph()
            elif len(items) == 1 or len(items) == 2 or len(items) == 3:
                terminate_states.append(int(items[0]))
            elif len(items) == 4:
                start, end, word, info = items
                assert not word.isspace()
                lmwt, acwt, _ = info.split(",", maxsplit=2)
                start, end = int(start), int(end)
                # The acwt in kaldi lattice is the negative logarithm
                lmwt = -float(lmwt) * lm_scale
                acwt = -float(acwt) * ac_scale
                word = word.lower() if lower_case else word
                G.add_edge(start, end, word=word, f=lmwt + acwt)
            elif line.isspace():
                if G is None:
                    continue
                for state in terminate_states:
                    G.add_edge(state, len(G), word=EOS, f=_EPS)
                terminate_states = []
                assert lattice_id not in lattices
                lattices[lattice_id] = lattice_to_line_graph(G) if to_line_graph else G
                lattice_id = G = None

    if G is not None:
        for state in terminate_states:
            G.add_edge(state, len(G), word=EOS, f=_EPS)
        assert lattice_id not in lattices
        lattices[lattice_id] = lattice_to_line_graph(G) if to_line_graph else G

    return lattices


LAT_INFO = namedtuple("LAT_INFO", ("words", "marginal", "min_depth", "max_depth", "fb", "mask"))


def line_graph_process(L: nx.DiGraph, use_marginal=True, min_depth=True, max_depth=True, forward_backward=True,
                       use_lattice_mask=True, binary_mask=True) -> LAT_INFO:
    """ Lattice -> LAT_INFO """
    assert all(isinstance(node, int) for node in L.nodes), "All node_id should be map to int"
    node_order = list(nx.topological_sort(L))

    # Add marginal / forward / backward score on graph
    L = scoring_line_graph(L, node_order)
    nodes = L.nodes

    words = []
    marginals = [] if use_marginal else None

    for i in range(len(L)):
        node_attr = nodes[i]
        words.append(node_attr["word"])
        if use_marginal:
            marginals.append(node_attr["m"])

    _min_depth, _max_depth = min_max_depth(L, node_order)
    _min_depth = [_min_depth[n] for n in range(len(L))] if min_depth else None
    _max_depth = [_max_depth[n] for n in range(len(L))] if max_depth else None
    fb_mtx = mtx_transfer(L) if forward_backward else None

    if use_lattice_mask:
        if binary_mask:
            lattice_mask = binary_reachability_mask(L)
        else:
            lattice_mask = probabilistic_reachability_mask(L)
    else:
        lattice_mask = None

    return LAT_INFO(" ".join(words), marginals, _min_depth, _max_depth, fb_mtx, lattice_mask)


_items_per_cache = 100000  # The number of pairs per cache file


def _load_cache(cache_prefix: str):
    if (cache_prefix is not None) and exists(cache_prefix + ".OK"):
        data = None
        for cache_file in sorted(glob.glob(cache_prefix + f'.*.pkl')):
            with open(cache_file, "rb") as f:
                _data = pickle.load(f)
            if data is None:
                data = type(_data)()
                logging.info(f"Cache format is {type(_data)}")
            if isinstance(_data, list):
                data.extend(_data)
            elif isinstance(_data, (set, dict)):
                data.update(_data)
            logging.info(f"Read cache from '{cache_file}'")
        return data
    return None


def _dump_cache(cache_prefix: str, data, items_per_cache):
    raw_type = type(data)
    if isinstance(data, list):
        data_items = data
    elif isinstance(data, set):
        data_items = list(data)
    elif isinstance(data, dict):
        data_items = list(data.items())
    else:
        raise ValueError(f"Unrecognized data type {type(data)}")

    for idx, i in zip(count(0, 1), range(0, len(data), items_per_cache)):
        cache_file = cache_prefix + f".{idx:03d}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(raw_type(data_items[i: i + items_per_cache]), f)
        logging.info(f"Cache data pairs to {cache_file}")

    open(cache_prefix + ".OK", "w")
    logging.info(f"Cache finished. Flag file '{cache_prefix}.OK' is saved.")


class _KeyBased:
    items: Dict[str, Any]

    def keys(self):
        yield from self.items.keys()

    def __getitem__(self, key):
        return self.items[key]

    def __len__(self):
        return len(self.items)


class _LatticeDataset(_KeyBased):
    """ Dataset to parse only kaldi's compact lattice files """

    def __init__(self, lattice_file_pattern: str, cache_prefix: str = None, min_nodes: int = 0,
                 max_nodes: int = MAX_NODEs, lower_case=True, lm_scale=1.0, ac_scale=1.0):
        # Try to read cache first
        if cache_prefix is not None:
            items = _load_cache(cache_prefix)
            if items is not None:
                self.items = items
                return

        self.items = {}  # type: Dict[str, List[nx.DiGraph]]

        # Read data
        files = sorted(glob.glob(lattice_file_pattern))
        if len(files) == 0:
            raise FileNotFoundError("No lattice files found")

        total_cnt = _EPS
        for file in files:
            _lattices = parse_lattice_file(file, to_line_graph=True, lower_case=lower_case,
                                           lm_scale=lm_scale, ac_scale=ac_scale)
            total_cnt += len(_lattices)
            logging.info(f"Load {len(_lattices)} lattices from {file}.")

            for lattice_id, lat in _lattices.items():
                if len(lat) < min_nodes or len(lat) > max_nodes:
                    continue
                if lattice_id in self.items:
                    self.items[lattice_id].append(lat)
                else:
                    self.items[lattice_id] = [lat]

        delta = total_cnt - len(self.items)
        logging.info(f"{self.__class__.__name__}: {len(self.items)} items. "
                     f"{delta} ({delta / total_cnt:.2%}) items filtered")

        # Cache data
        if cache_prefix is not None:
            _dump_cache(cache_prefix, self.items, _items_per_cache)


class __TextDataset(_KeyBased):
    """ Dataset to parse only text file """

    def __init__(self, text_file_pattern: str, lower_case=True, token_split=False):
        self.items = {}

        # Read data
        files = sorted(glob.glob(text_file_pattern))
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
        with open(text_file, encoding="utf8") as f:
            for line in f:
                if len(line) < 1:
                    continue
                _id, tokens = line.split(maxsplit=1)
                if lower_case:
                    tokens = tokens.lower()
                if token_split:
                    tokens = tokens.split()
                items.append((_id, tokens))
        return items


class _OracleDataset(__TextDataset):
    def __init__(self, text_file_pattern: str, lower_case=True):
        super().__init__(text_file_pattern, lower_case=lower_case, token_split=True)


class _TextDataset(__TextDataset):
    def __init__(self, text_file_pattern: str, lower_case=True):
        super().__init__(text_file_pattern, lower_case=lower_case, token_split=False)


class LatticeTextDataset(Dataset):
    def __init__(
            self, lattice_pattern: str, text_pattern: str = None, cache_prefix: str = None,
            min_nodes=0, max_nodes=MAX_NODEs, lower_case=True, lm_scale=1.0, ac_scale=1.0,
            use_marginal=True, use_min_depth=True, use_max_depth=True, use_forward_backward=True,
            use_lattice_mask=True, binary_mask=True, sort_by_len=True
    ):
        global WARNED
        if WARNED:
            WARNED = False
            logging.warning("计算forward分数时要不要加上语言模型的分数？")
            logging.warning("声学模型分数的权重怎么设置？")

        self.use_marginal = use_marginal
        self.use_min_depth = use_min_depth
        self.use_max_depth = use_max_depth
        self.use_forward_backward = use_forward_backward
        self.use_lattice_mask = use_lattice_mask
        self.binary_mask = binary_mask

        lattices = _LatticeDataset(lattice_pattern, cache_prefix=cache_prefix, min_nodes=min_nodes, max_nodes=max_nodes,
                                   lower_case=lower_case, lm_scale=lm_scale, ac_scale=ac_scale)
        keys = set(lattices.keys())

        if text_pattern is None:
            self.items = [[key, lattices[key], None] for key in sorted(keys)]
        else:
            transcripts = _TextDataset(text_pattern, lower_case=lower_case)
            keys = keys & set(transcripts.keys())
            self.items = [[key, lattices[key], transcripts[key]] for key in sorted(keys)]

        if sort_by_len:
            self.sort_by_length()

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

    def partial_collate_fn(self, tokenizer: BartTokenizer, n_lat=1):

        def padding_seqs(seqs: list, pad, max_len, tensor_type):
            for i, seq in enumerate(seqs):
                seqs[i] = seq + (max_len - len(seq)) * [pad]
            return torch.tensor(seqs, dtype=tensor_type)

        def padding_mats(mats: list, pad, shape, tensor_type):
            _mats = torch.zeros(len(mats), *shape, dtype=tensor_type).fill_(pad)
            for i, mat in enumerate(mats):
                m, n = mat.shape
                _mats[i, :m, :n] = mat
            return _mats

        def collate_fn(batch_data: list) -> dict:
            # TODO: 当num_workers > 0时，collate_fn里的修改会不会反映到主进程上？
            batch_input_seqs = []
            batch_min_depth = []
            batch_max_depth = []
            batch_lat_mask = []
            batch_fb = []
            batch_marginal = []

            batch_texts = []

            for _key, lats, text in batch_data:
                idx = [i % len(lats) for i in range(max(len(lats), n_lat))]  # Padding lats if not enough
                random.shuffle(idx)
                idx = idx[:n_lat]
                batch_texts.extend([text] * n_lat)

                # Process nx.Digraph-formatted lattice
                for i in idx:
                    lat = lats[i]
                    if isinstance(lat, nx.DiGraph):
                        lats[i] = lat = line_graph_process(
                            lat, use_marginal=self.use_marginal, min_depth=self.use_min_depth,
                            max_depth=self.use_max_depth, forward_backward=self.use_forward_backward,
                            use_lattice_mask=self.use_lattice_mask, binary_mask=self.binary_mask
                        )
                    input_text, marginal, min_depth, max_depth, fb, lattice_mask = lat
                    batch_input_seqs.append(input_text)
                    batch_min_depth.append(min_depth)
                    batch_max_depth.append(max_depth)
                    batch_lat_mask.append(lattice_mask)
                    batch_fb.append(fb)
                    batch_marginal.append(marginal)

            inputs = word_based_tokenize(batch_input_seqs, tokenizer, lower=True,
                                         return_attention=False, to_tensor=True)

            B, max_len = inputs["input_ids"].shape
            mat_shape = (max_len, max_len)

            inputs["min_depth"] = None if batch_min_depth[0] is None else \
                padding_seqs(batch_min_depth, MAX_NODEs, max_len, torch.long)
            inputs["max_depth"] = None if batch_max_depth[0] is None else \
                padding_seqs(batch_max_depth, MAX_NODEs, max_len, torch.long)
            inputs["lattice_mask"] = None if batch_lat_mask[0] is None else \
                padding_mats(batch_lat_mask, 0, mat_shape, torch.long)
            inputs["marginals"] = None if batch_marginal[0] is None else \
                padding_seqs(batch_marginal, 0, max_len, torch.float)
            inputs["forward_backward"] = None if batch_fb[0] is None else \
                padding_mats(batch_fb, 0, mat_shape, torch.float)
            inputs["batch_idx"] = torch.arange(B, dtype=torch.long)

            if batch_texts[0] is not None:
                outputs = tokenizer.batch_encode_plus(batch_texts, padding=True, return_tensors="pt")
                inputs["decoder_input_ids"] = outputs["input_ids"][:, :-1].contiguous()
                inputs["decoder_attention_mask"] = outputs["attention_mask"][:, :-1].contiguous()
                inputs["labels"] = outputs["input_ids"][:, 1:].contiguous()

            return inputs

        return collate_fn
