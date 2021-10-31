import json
from typing import Union

import networkx as nx
import numpy as np


def dump_graph(G: nx.DiGraph, filename: str = None) -> dict:
    graph = {
        "nodes": [(n, G.nodes[n]) for n in G.nodes],
        "edges": [(e, G.edges[e]) for e in G.edges]
    }
    if filename is not None:
        with open(filename, "w", encoding="utf8") as f:
            json.dump(graph, f, indent=1)
    return graph


def load_graph(graph: Union[dict, str]) -> nx.DiGraph:
    if isinstance(graph, str):
        with open(graph, "r", encoding="utf8") as f:
            graph = json.load(f)
    elif not isinstance(graph, dict):
        raise ValueError(f"Unrecognized type of graph ({type(graph)})")

    G = nx.DiGraph()
    G.add_nodes_from(graph['nodes'])
    G.add_edges_from(graph['edges'])
    return G


def log_sum_exp(logp: np.ndarray):
    logp = logp - max(logp)
    return np.log(np.sum(np.exp(logp)))
