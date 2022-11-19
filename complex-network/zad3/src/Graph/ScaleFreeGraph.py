import networkx as nx

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class ScaleFreeGraph(NxGraphAnimator):
    def __init__(self, nodes_count: int = 8, edges_count: int = 4,
                 layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        super().__init__(layout)

        self.nodes_count = nodes_count
        self.edges_count = edges_count

        self.initialise_new_random_graph()

    def initialise_new_random_graph(self):
        self.graph = nx.barabasi_albert_graph(self.nodes_count, self.edges_count)

    def save_visualisation_to_file(self, filename: str = None):
        if filename is None:
            filename = f'{self.default_filepath_images}/scale_free_graph.png'

        super().save_visualisation_to_file(filename)

    def save_animation_to_file(self, filename: str = None):
        if filename is None:
            filename = f'{self.default_filepath_animations}/scale_free_graph_animation.gif'

        super().save_animation_to_file(filename)
