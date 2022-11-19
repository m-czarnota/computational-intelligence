import networkx as nx
import numpy as np
from matplotlib import animation, rc, pyplot as plt

from NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class NxGraphAnimator:
    def __init__(self, layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        self.layout = NxGraphAnimatorPosLayoutEnum.get_nx_layout(layout)

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
        if self.actual_node < len(self.nodes_to_draw):
            self.draw_node__()
            return

        if self.actual_edge < len(self.edges_to_draw):
            self.draw_edge__()
            return

    def draw_node__(self):
        self.graph.add_node(self.nodes_to_draw[self.actual_node])
        self.actual_node += 1

        nx.draw(self.graph, with_labels=True, pos=self.pos)

    def draw_edge__(self):
        self.graph.add_edge(self.edges_to_draw[self.actual_edge][0], self.edges_to_draw[self.actual_edge][1],
                            weight=0.9)
        self.actual_edge += 1

        nx.draw(self.graph, with_labels=True, pos=self.pos)
