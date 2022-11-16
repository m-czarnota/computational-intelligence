import networkx as nx
import numpy as np
from matplotlib import animation, rc, pyplot as plt
from enum import Enum


def random_graph():
    node_count = 8
    k = 2

    g = nx.erdos_renyi_graph(node_count, k)

    nx.draw(g, with_labels=True)
    plt.show()

    animator = NxGraphAnimator()
    animator.draw(g)


def small_world_graph():
    nodes_count = 8
    k = 3
    p = 1

    g = nx.watts_strogatz_graph(nodes_count, k, p)
    pos = nx.circular_layout(g)

    nx.draw(g, pos=pos, with_labels=True)
    plt.show()

    animator = NxGraphAnimator(NxGraphAnimatorPosLayoutEnum.CIRCULAR)
    animator.draw(g)


def without_scaling_graph():
    nodes_count = 8
    edges_count = 4

    g = nx.barabasi_albert_graph(nodes_count, edges_count)

    nx.draw(g, with_labels=True)
    plt.show()


class NxGraphAnimatorPosLayoutEnum(Enum):
    SPRING = nx.spring_layout
    CIRCULAR = nx.circular_layout
    PLANAR = nx.planar_layout
    SHELL = nx.shell_layout

    @classmethod
    def get_nx_layout(cls, value):
        if value == cls.SPRING:
            return nx.spring_layout



class NxGraphAnimator:
    def __init__(self, layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        self.layout = layout

        self.graph = None
        self.nodes_to_draw = None
        self.edges_to_draw = None
        self.actual_node = None
        self.actual_edge = None

        self.fig = None
        self.pos = None

    def draw(self, graph):
        self.actual_node = 0
        self.actual_edge = 0

        self.graph = nx.Graph()
        self.fig = plt.figure()
        self.pos = self.layout(graph)

        self.nodes_to_draw = np.array(graph.nodes())
        self.edges_to_draw = np.array(graph.edges())

        ani = animation.FuncAnimation(self.fig, self.update_animation__, repeat=False)
        rc('animation', html='jshtml')
        plt.show()

    def update_animation__(self, frame):
        self.fig.clear()

        if self.actual_node < len(self.nodes_to_draw):
            self.graph.add_node(self.nodes_to_draw[self.actual_node])
            self.actual_node += 1

            nx.draw(self.graph, with_labels=True, pos=self.pos)

            return

        self.graph.add_edge(self.edges_to_draw[self.actual_edge][0], self.edges_to_draw[self.actual_edge][1], weight=0.9)
        self.actual_edge += 1

        nx.draw(self.graph, with_labels=True, pos=self.pos)


if __name__ == '__main__':
    small_world_graph()
