import networkx as nx
from matplotlib import pyplot as plt

if __name__ == '__main__':
    G = nx.complete_graph(100)  # example graph

    edges = nx.read_weighted_edgelist('rt_assad.edges', delimiter=',')
    dolphins = nx.read_edgelist('soc-dolphins.mtx')
    caltech = nx.read_edgelist('socfb-Caltech36.mtx')

    print(edges)
    print(dolphins.nodes)
    print(caltech)

    dolphins.remove_node('11')
    print(dolphins.nodes)

    dolphins.add_node("106")
    dolphins.nodes['106']['name'] = 'val'
    dolphins.nodes['106']['name'] = 'val'
    print(dolphins.nodes['106'])

    print(dolphins.edges)

    colors = ['blue' if int(node) <= 40 else 'green' for node in dolphins]
    colors_edges = ['r' if int(v) <= 40 else 'y' for u, v in dolphins.edges]
    pos = nx.spring_layout(dolphins)

    nx.draw_networkx_nodes(dolphins, pos=pos, node_color=colors)
    nx.draw_networkx_edges(dolphins, pos=pos, edge_color=colors_edges)
    nx.draw_networkx_labels(dolphins, pos=pos)
    plt.show()
