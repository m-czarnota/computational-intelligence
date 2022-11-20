import networkx as nx
import numpy as np
from matplotlib import animation, rc, pyplot as plt
from abc import ABC, abstractmethod

from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class NxGraphAnimator(ABC):
    def __init__(self, layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        self.layout = NxGraphAnimatorPosLayoutEnum.get_nx_layout(layout)

        self.graph = None
        self.nodes_to_draw = None
        self.edges_to_draw = None
        self.actual_node = None
        self.actual_edge = None

        self.fig = None
        self.pos = None

        self.default_filepath_images = './images'
        self.default_filepath_animations = './animations'

        self.graph_to_animate__ = None
        self.default_filename__ = None

    @abstractmethod
    def initialise_new_random_graph(self):
        ...

    def save_visualisation_to_file(self, filename: str = None):
        if self.handle_can_visualise__() is False:
            return

        if filename is None:
            filename = f'{self.default_filepath_images}/{self.default_filename__}.png'

        self.visualise__()
        plt.savefig(filename)

    def save_animation_to_file(self, filename: str = None):
        if self.handle_can_visualise__() is False:
            return

        if filename is None:
            filename = f'{self.default_filepath_animations}/{self.default_filename__}_animation.gif'

        anim = self.draw_animate__()
        anim.save(filename)

    def show_animation(self):
        if self.handle_can_visualise__() is False:
            return

        anim = self.draw_animate__()
        plt.show()

    def show_graph(self):
        if self.handle_can_visualise__() is False:
            return

        self.visualise__()
        plt.show()

    def handle_can_visualise__(self) -> bool:
        if self.graph is None:
            print('Graph is empty! You have to initialise a graph.')
            return False

        return True

    def draw_animate__(self):
        if self.handle_can_visualise__() is False:
            return

        # plt.clf()
        self.prepare_to_draw__()

        anim = animation.FuncAnimation(self.fig, self.update_animation__, repeat=False)
        rc('animation', html='jshtml')

        return anim

    def visualise__(self):
        plt.clf()
        pos = self.layout(self.graph)
        nx.draw(self.graph, pos=pos, with_labels=True)

    def prepare_to_draw__(self):
        self.actual_node = 0
        self.actual_edge = 0

        self.graph_to_animate__ = nx.Graph()
        self.fig = plt.figure()
        self.pos = self.layout(self.graph)

        self.nodes_to_draw = np.array(self.graph.nodes())
        self.edges_to_draw = np.array(self.graph.edges())

    def update_animation__(self, frame):
        if self.actual_node < len(self.nodes_to_draw):
            self.draw_node__()
            return

        if self.actual_edge < len(self.edges_to_draw):
            self.draw_edge__()
            return

    def draw_node__(self):
        self.graph_to_animate__.add_node(self.nodes_to_draw[self.actual_node])
        self.actual_node += 1

        nx.draw(self.graph_to_animate__, with_labels=True, pos=self.pos)

    def draw_edge__(self):
        self.graph_to_animate__.add_edge(self.edges_to_draw[self.actual_edge][0],
                                         self.edges_to_draw[self.actual_edge][1],
                                         weight=0.9)
        self.actual_edge += 1

        nx.draw(self.graph_to_animate__, with_labels=True, pos=self.pos)
