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


def display_lattice(G: nx.DiGraph):
    pos = _lattice_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='#66b2ff', node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True)

    nx.draw_networkx_labels(G, pos)

    edge_labels = {}
    for edge in G.edges:
        info = G[edge]
        # info = {k: f"{v:.1e}"for k, v in info.items()}
        edge_labels[edge] = info
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, rotate=False)

    plt.show()


def display_line_graph(L: nx.DiGraph, title: str = None):
    pos = _lattice_layout(L)

    node_color = [L.nodes[n]["m"] for n in L]
    nx.draw_networkx_nodes(L, pos, node_color=node_color, node_size=500)
    nx.draw_networkx_edges(L, pos, arrows=True)

    node_labels = {}
    for node in L.nodes:
        node_labels[node] = L.nodes[node]['word']
    nx.draw_networkx_labels(L, pos, labels=node_labels, font_size=10)

    edge_labels = {}
    for edge in L.edges:
        info = L[edge[0]][edge[1]]
        info = {k: f"{v:.1e}" for k, v in info.items()}
        edge_labels[edge] = info
    nx.draw_networkx_edge_labels(L, pos, edge_labels=edge_labels, font_size=10, rotate=False)

    if title is not None:
        plt.title(title)
    plt.show()
