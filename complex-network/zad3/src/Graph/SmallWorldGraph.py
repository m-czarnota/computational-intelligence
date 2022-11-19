import networkx as nx
from matplotlib import pyplot as plt

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class SmallWorldGraph(NxGraphAnimator):
    def __init__(self, nodes_count: int = 8, nearest_neighbour_count: int = 4, each_rewiring_prob: float = 0.7,
                 layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.CIRCULAR):
        super().__init__(layout)

        self.nodes_count = nodes_count
        self.nearest_neighbour_count = nearest_neighbour_count
        self.each_rewiring_prob = each_rewiring_prob

        self.initialise_new_random_graph()

    def initialise_new_random_graph(self):
        self.graph = nx.watts_strogatz_graph(self.nodes_count, self.nearest_neighbour_count, self.each_rewiring_prob)

    def save_visualisation_to_file(self, filename: str = None):
        if filename is None:
            filename = f'{self.default_filepath_images}/small_world_graph.png'

        super().save_visualisation_to_file(filename)

    def save_animation_to_file(self, filename: str = None):
        if filename is None:
            filename = f'{self.default_filepath_animations}/small_world_graph_animation.gif'

        super().save_animation_to_file(filename)
