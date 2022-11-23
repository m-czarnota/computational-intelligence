import networkx as nx

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.Enum.NxGraphAnimatorPosLayoutEnum import NxGraphAnimatorPosLayoutEnum


class RandomGraph(NxGraphAnimator):
    def __init__(self, node_count: int = 8, edge_creation_prob: float = 0.6,
                 layout: NxGraphAnimatorPosLayoutEnum = NxGraphAnimatorPosLayoutEnum.SPRING):
        super().__init__(layout)
        self.node_count = node_count
        self.edge_creation_prob = edge_creation_prob

        self.default_filename__ = 'random_graph'

        self.initialise_new_random_graph__()

    def initialise_new_random_graph__(self):
        self.graph = nx.erdos_renyi_graph(self.node_count, self.edge_creation_prob)
