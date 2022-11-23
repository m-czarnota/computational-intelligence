import networkx as nx

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class ScaleFreeGraph(NxGraphAnimator):
    def __init__(self, nodes_count: int = 8, edges_count: int = 4,
                 layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        super().__init__(layout)

        self.nodes_count = nodes_count
        self.edges_count = edges_count

        self.default_filename__ = 'scale_free_graph'

        self.initialise_new_random_graph__()

    def initialise_new_random_graph__(self):
        self.graph = nx.barabasi_albert_graph(self.nodes_count, self.edges_count)
