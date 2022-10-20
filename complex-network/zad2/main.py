import random
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection


def update(t, number_of_nodes: int, draw_nodes: PathCollection):
    nc = [[0.9, 0.9, 0.9]] * number_of_nodes
    random_node_number = random.randint(0, number_of_nodes - 1)
    nc[random_node_number] = [1, 0.5, 0.5]
    draw_nodes.set_color(nc)

    return draw_nodes,


def update2(t, draw_nodes: PathCollection, nodes, visited):
    number_of_nodes = len(nodes)
    nc = [[0.9, 0.9, 0.9]] * number_of_nodes

    for i, node in enumerate(nodes):
        # print(node)
        if i == t and visited[i] is False:
            nc[i] = [0.5, 0.5, 1]
            visited[t] = True
        elif i == t and visited[i] is True:
            visited[t] = False

        if i == t and 'name' in nodes[node].keys():
            nc[i] = [1, 0.5, 0.5]

    draw_nodes.set_color(nc)

    return draw_nodes,


if __name__ == '__main__':
    G = nx.complete_graph(20)
    G.add_node("106")
    G.nodes['106']['name'] = 'val'

    number_of_nodes = G.number_of_nodes()
    nc = [[0.9, 0.9, 0.9]] * number_of_nodes

    fig = plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G)

    draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=nc)
    edges = nx.draw_networkx_edges(G, pos)
    labels = {i: f'{i + 1}' for i in range(number_of_nodes)}
    # nx.draw_networkx_labels(G, pos, labels, font_size=12)
    print(G.nodes)
    for node in G.nodes:
        print(node)

    visited = [False for i in range(number_of_nodes)]
    anim = FuncAnimation(fig, update2, fargs=(draw_nodes, G.nodes, visited), interval=400)
    # plt.close()
    rc('animation', html='jshtml')
    plt.show()
