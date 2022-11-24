import networkx as nx

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class SmallWorldGraph(NxGraphAnimator):
    def __init__(self, nodes_count: int = 8, nearest_neighbour_count: int = 4, each_rewiring_prob: float = 0.7,
                 layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.CIRCULAR):
        super().__init__(layout)

        self.nodes_count = nodes_count
        self.nearest_neighbour_count = nearest_neighbour_count
        self.each_rewiring_prob = each_rewiring_prob

        self.default_filename__ = 'small_world_graph'

        self.initialise_new_random_graph__()

    def initialise_new_random_graph__(self):
        self.graph = nx.watts_strogatz_graph(self.nodes_count, self.nearest_neighbour_count, self.each_rewiring_prob)
