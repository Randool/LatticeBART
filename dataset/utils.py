import json
from typing import Union

import networkx as nx


def dump_graph(G: nx.DiGraph, filename: str = None) -> dict:
    graph = {
        "nodes": [(n, G.nodes[n]) for n in G.nodes],
        "edges": [(u, v, G[u][v]) for u, v in G.edges]
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
