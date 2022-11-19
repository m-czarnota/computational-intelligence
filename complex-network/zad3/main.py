import networkx as nx
from matplotlib import pyplot as plt

from NxGraphAnimator import NxGraphAnimator
from NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


def random_graph():
    node_count = 8
    edge_creation_prob = 0.6

    g = nx.erdos_renyi_graph(node_count, edge_creation_prob)
    pos = nx.spring_layout(g)

    nx.draw(g, pos=pos, with_labels=True)
    plt.show()

    animator = NxGraphAnimator()
    animator.draw(g)


def small_world_graph():
    nodes_count = 8
    nearest_neighbour_count = 4
    each_rewiring_prob = 0.7

    g = nx.watts_strogatz_graph(nodes_count, nearest_neighbour_count, each_rewiring_prob)
    pos = nx.circular_layout(g)

    nx.draw(g, pos=pos, with_labels=True)
    plt.show()

    animator = NxGraphAnimator(NxGraphAnimatorPosLayoutEnum.CIRCULAR)
    animator.draw(g)


def without_scaling_graph():
    nodes_count = 8
    edges_count = 4

    g = nx.barabasi_albert_graph(nodes_count, edges_count)
    pos = nx.spring_layout(g)

    nx.draw(g, pos=pos, with_labels=True)
    plt.show()

    animator = NxGraphAnimator()
    animator.draw(g)


if __name__ == '__main__':
    random_graph()
    small_world_graph()
    without_scaling_graph()
