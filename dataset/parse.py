import logging
import sys
from collections import namedtuple
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import torch

from dataset.symbols import *
from dataset.utils import dump_graph

_EPS = sys.float_info.min


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

    # Replacing eos node
    eos_edges = [edge for edge in L.edges if L.nodes[edge[1]]['word'] == EOS]

    if len(eos_edges) == 0:
        logging.warning(f"NO `eos_edges` found!! Simply add edge towards EOS. Graph info: {dump_graph(L)}")
        eos_node = (0, 1)
        L.add_node(eos_node, word=EOS)
        L.add_edge((0, 0), eos_node, f=_EPS)
    else:
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


def binary_reachability_mask(L: nx.DiGraph):
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


def __parse_lattice(lines, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) -> Tuple[str, nx.DiGraph]:
    lattice_id = G = None
    terminate_states = []
    terminate_wt = []

    for line in lines:
        if len(line.strip()) == 0:
            break

        items = line.strip().split()

        if len(items) == 1 and lattice_id is None:
            lattice_id = items[0]
            G = nx.DiGraph()

        elif len(items) == 1:
            terminate_states.append(int(items[0]))
            terminate_wt.append(lm_scale + ac_scale)

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

            G.add_edge(int(start_state), int(end_state), word=word, f=lmwt + acwt)

    assert lattice_id is not None
    assert G is not None

    end_state = len(G)

    for state, wt in zip(terminate_states, terminate_wt):
        G.add_edge(state, end_state, word=EOS, f=wt)

    G = lattice_to_line_graph(G) if to_line_graph else G

    return lattice_id, G


def _parse_lattice(lines, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) -> Tuple[str, nx.DiGraph]:
    try:
        return __parse_lattice(lines, to_line_graph, lower_case, lm_scale, ac_scale)
    except Exception:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        print("\n".join(lines), file=sys.stderr)
        sys.exit(-1)


def parse_lattice_file(lattice_file: str, to_line_graph=True, lower_case=True, lm_scale=1.0, ac_scale=1.0) \
        -> Dict[str, nx.DiGraph]:
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
        -> Tuple[str, nx.DiGraph]:
    filename, bias, delta = lattice_addr.split(":")

    with open(filename, "r", encoding="utf8") as f:
        f.seek(int(bias))
        lines = f.read(int(delta)).split("\n")

    return _parse_lattice(lines, to_line_graph=to_line_graph, lower_case=lower_case,
                          lm_scale=lm_scale, ac_scale=ac_scale)


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
