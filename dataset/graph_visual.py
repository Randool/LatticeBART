import copy

import matplotlib.pyplot as plt
import networkx as nx

from dataset.parse import min_max_depth


def _lattice_layout(G: nx.DiGraph):
    pos = nx.spring_layout(G)
    _, depths = min_max_depth(G)
    for _id in depths.keys():
        depth = depths[_id]
        pos[_id][1] = depth * -1
    return pos


def display_lattice(G: nx.MultiDiGraph):
    pos = _lattice_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='#66b2ff', node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True)

    nx.draw_networkx_labels(G, pos)

    # edge_labels = {}
    # edges = G.edges
    # for edge in edges:
    #     info = edges[edge]
    #     # info = {k: f"{v:.1e}"for k, v in info.items()}
    #     edge_labels[edge] = info

    # edge_labels = nx.get_edge_attributes(G, "word")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, rotate=False)
    plt.show()


def display_line_graph(L: nx.MultiDiGraph, title: str = None):
    _L = nx.DiGraph(copy.deepcopy(L))
    pos = _lattice_layout(_L)
    nodes = _L.nodes
    node_color = [nodes[n]["log_m"] for n in _L]
    nx.draw_networkx_nodes(_L, pos, node_color=node_color, node_size=500)
    nx.draw_networkx_edges(_L, pos, arrows=True)

    node_labels = {}
    for node in nodes:
        node_labels[node] = nodes[node]['word']
    nx.draw_networkx_labels(_L, pos, labels=node_labels, font_size=10)

    edge_labels = {}
    for edge in _L.edges:
        info = _L[edge[0]][edge[1]]
        info = {k: f"{v:.1e}" for k, v in info.items()}
        edge_labels[edge] = info
    nx.draw_networkx_edge_labels(_L, pos, edge_labels=edge_labels, font_size=10, rotate=False)

    if title is not None:
        plt.title(title)
    plt.show()
