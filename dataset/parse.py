import sys
from collections import namedtuple
from itertools import accumulate
from math import exp
from typing import Dict, Tuple, List

import networkx as nx
import numpy as np
import torch
from scipy.special import logsumexp
from transformers import BartTokenizer

from dataset.symbols import *

_EPS = sys.float_info.min


def lattice_to_line_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """ Convert node-labeled lattice to edge-labeled lattices """
    L = nx.line_graph(G)  # type: nx.MultiDiGraph
    bos_for_L = (0, 0, 0)
    L.add_node(bos_for_L, word=BOS)

    for edge in G.edges:
        attrs = G.edges[edge]
        word = attrs.pop("word")
        nx.set_node_attributes(L, {edge: {"word": word}})
        if edge[0] == 0:
            L.add_edge(bos_for_L, edge, **attrs)
        else:
            for pred in L.predecessors(edge):
                nx.set_edge_attributes(L, {(pred, edge, 0): attrs})

    # Replacing eos node
    eos_nodes = [node for node in L.nodes if L.out_degree[node] == 0]
    eos_node = eos_nodes[0]

    for node in eos_nodes:
        if node == eos_node:
            continue
        for pred in L.predecessors(node):
            attr = L.edges[(pred, node, 0)]
            L.add_edge(pred, eos_node, **attr)
        L.remove_node(node)

    # Rename node_id
    rename = {j: i for i, j in enumerate(nx.topological_sort(L))}
    L = nx.relabel_nodes(L, rename, copy=False)
    return L


def scoring_line_graph(L: nx.MultiDiGraph, topo_order: list = None) -> nx.MultiDiGraph:
    """ Assign marginal/forward/backward score on the given line graph """
    if topo_order is None:
        topo_order = list(nx.topological_sort(L))
    nodes, edges = L.nodes, L.edges

    # Forward scores and marginal scores can be calculated simultaneously
    nx.set_node_attributes(L, -float('inf'), "log_m")
    nodes[topo_order[0]]['log_m'] = 0.0

    for i in topo_order:
        log_m_i = nodes[i]["log_m"]
        succ_nodes = list(L.successors(i))
        if len(succ_nodes) == 0:
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
        log_fs = np.array([edges[(i, j, 0)]["log_f"] for j in succ_nodes])
        log_fs = (log_fs - logsumexp(log_fs)).tolist()

        for j, log_f in zip(succ_nodes, log_fs):
            edges[(i, j, 0)]["log_f"] = log_f
            nodes[j]["log_m"] = logsumexp([nodes[j]["log_m"], log_m_i + log_f])

    # Calculate backward scores
    for edge in edges:
        edges[edge]["log_b"] = edges[edge]["log_f"] + nodes[edge[0]]["log_m"] - nodes[edge[1]]["log_m"]

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


def mtx_relative_position(L: nx.MultiDiGraph, token_len: List[int] = None) -> torch.Tensor:
    assert all(isinstance(node, int) for node in L.nodes), "All node_id should be map to int"

    if token_len is None:
        token_len = [1] * len(L)
    _len = sum(token_len)
    start_id = [0] + list(accumulate(token_len))

    distance = nx.all_pairs_shortest_path_length(L)
    mtx = torch.zeros(_len, _len, dtype=torch.int).fill_(torch.iinfo(torch.int).min)
    for i, succs in distance:
        for j, d in succs.items():
            i_start, i_end = start_id[i], start_id[i + 1]
            j_start, j_end = start_id[j], start_id[j + 1]
            mtx[i_start: i_end, j_start: j_end] = d
            mtx[j_start: j_end, i_start: i_end] = -d

    return mtx


def binary_reachability_mask(L: nx.MultiDiGraph, token_len: List[int] = None):
    relation = mtx_relative_position(L, token_len)
    return relation.ne(torch.iinfo(torch.int).min).long()


def mtx_transfer(L: nx.MultiDiGraph, token_len: List[int] = None) -> torch.Tensor:
    if token_len is None:
        token_len = [1] * len(L)
    _len = sum(token_len)
    mtx = torch.zeros(_len, _len, dtype=torch.float)
    start_id = [0] + list(accumulate(token_len))

    for i in L:
        for j, attrs in L[i].items():
            # print(i, j, attrs)
            i_start, i_end = start_id[i], start_id[i + 1]
            j_start, j_end = start_id[j], start_id[j + 1]
            mtx[i_start: i_end, j_start: j_end] = attrs[0]["log_f"]
            mtx[j_start: j_end, i_start: i_end] = attrs[0]["log_b"]

    return mtx


def __parse_lattice(lines, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) \
        -> Tuple[str, nx.MultiDiGraph]:
    lattice_id = G = None
    terminate_states = []
    terminate_wt = []

    for line in lines:
        if len(line.strip()) == 0:
            break

        items = line.strip().split()

        if len(items) == 1 and lattice_id is None:
            lattice_id = items[0]
            G = nx.MultiDiGraph()

        elif len(items) == 1:
            terminate_states.append(int(items[0]))
            terminate_wt.append(0)

        elif len(items) == 2:
            terminate_states.append(int(items[0]))
            lmwt, acwt, *_ = items[1].split(",", maxsplit=2)
            terminate_wt.append(-float(lmwt) * lm_scale + -float(acwt) * ac_scale)

        else:  # len(items) == 3 or 4
            if len(items) == 3:  # 6 7 right
                start_state, end_state, word = items
                lmwt, acwt = 1, 1
            else:
                start_state, end_state, word, info = items
                lmwt, acwt, _ = info.split(",", maxsplit=2)

            # The acwt in kaldi lattice is the negative logarithm
            lmwt = -float(lmwt) * lm_scale
            acwt = -float(acwt) * ac_scale

            assert not word.isspace()
            word = word.lower() if lower_case else word

            G.add_edge(int(start_state), int(end_state), word=word, log_f=lmwt + acwt)

    assert lattice_id is not None
    assert G is not None

    end_state = len(G)

    for state, wt in zip(terminate_states, terminate_wt):
        G.add_edge(state, end_state, word=EOS, log_f=wt)

    if to_line_graph:
        G = lattice_to_line_graph(G)

    return lattice_id, G


def _parse_lattice(lines, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) \
        -> Tuple[str, nx.MultiDiGraph]:
    try:
        return __parse_lattice(lines, to_line_graph, lower_case, lm_scale, ac_scale)
    except Exception:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        print("\n".join(lines), file=sys.stderr)
        sys.exit(-1)


def parse_lattice_file(lattice_file: str, to_line_graph=True, lower_case=True, lm_scale=1.0,
                       ac_scale=1.0) -> Dict[str, nx.MultiDiGraph]:
    """ Parse lattice file and return line-graphs """
    lattices = {}
    lines = []

    with open(lattice_file, "rt", encoding="utf8") as f:
        for line in f:
            if line.isspace():
                if len(lines):
                    lattice_id, G = _parse_lattice(lines, to_line_graph=to_line_graph, lower_case=lower_case,
                                                   lm_scale=lm_scale, ac_scale=ac_scale)
                    lattices[lattice_id] = G
                    lines = []
            else:
                lines.append(line)

    if len(lines):
        lattice_id, G = _parse_lattice(lines, to_line_graph=to_line_graph, lower_case=lower_case,
                                       lm_scale=lm_scale, ac_scale=ac_scale)
        lattices[lattice_id] = G

    return lattices


def parse_lattice_from_address(lattice_addr: str, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) \
        -> Tuple[str, nx.MultiDiGraph]:
    filename, bias, delta = lattice_addr.split(":")

    with open(filename, "r", encoding="utf8") as f:
        f.seek(int(bias))
        lines = f.read(int(delta)).split("\n")

    return _parse_lattice(lines, to_line_graph=to_line_graph, lower_case=lower_case,
                          lm_scale=lm_scale, ac_scale=ac_scale)


LAT_INFO = namedtuple("LAT_INFO", ("token_ids", "marginal", "min_depth", "max_depth", "fb", "mask"))


def line_graph_process(L: nx.MultiDiGraph, tokenizer: BartTokenizer, use_marginal=True, use_min_depth=True,
                       use_max_depth=True, use_forward_backward=True, use_lattice_mask=True) -> LAT_INFO:
    """ 将Digraph转为LAT_INFO，并且包含了token的展开 """
    assert all(isinstance(node, int) for node in L.nodes), "All node_id should be map to int"
    node_order = list(nx.topological_sort(L))

    # Add marginal / forward / backward score on graph
    L = scoring_line_graph(L, node_order)
    nodes = L.nodes

    token_ids = []
    token_len = []
    marginals = [] if use_marginal else None
    min_depth = [] if use_min_depth else None
    max_depth = [] if use_max_depth else None

    _min_depth, _max_depth = min_max_depth(L, node_order) if (use_min_depth and use_max_depth) else (None, None)

    for n in range(len(L)):
        node_attr = nodes[n]
        word = node_attr["word"]
        ids = tokenizer.encode(word, add_prefix_space=word not in special_tokens, add_special_tokens=False)
        token_ids.extend(ids)
        _len = len(ids)
        token_len.append(_len)

        if use_marginal:
            marginals.extend([exp(node_attr["log_m"])] * _len)
        if use_min_depth:
            min_depth.extend([_min_depth[n]] * _len)
        if use_max_depth:
            max_depth.extend([_max_depth[n]] * _len)

    fb_mtx = torch.exp(mtx_transfer(L, token_len)) if use_forward_backward else None
    lattice_mask = binary_reachability_mask(L, token_len) if use_lattice_mask else None

    return LAT_INFO(token_ids, marginals, min_depth, max_depth, fb_mtx, lattice_mask)
