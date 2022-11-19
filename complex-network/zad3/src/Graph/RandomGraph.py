import networkx as nx
from matplotlib import pyplot as plt

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class RandomGraph(NxGraphAnimator):
    def __init__(self, node_count: int = 8, edge_creation_prob: float = 0.6,
                 layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        super().__init__(layout)
        self.node_count = node_count
        self.edge_creation_prob = edge_creation_prob

        self.initialise_new_random_graph()

    def initialise_new_random_graph(self):
        self.graph = nx.erdos_renyi_graph(self.node_count, self.edge_creation_prob)

    def save_visualisation_to_file(self, filename: str = None):
        if filename is None:
            filename = f'{self.default_filepath_images}/random_graph.png'

        super().save_visualisation_to_file(filename)

    def save_animation_to_file(self, filename: str = None):
        if filename is None:
            filename = f'{self.default_filepath_animations}/random_graph_animation.gif'

        super().save_animation_to_file(filename)
