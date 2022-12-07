import numpy as np

from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.DTO.NodeDto import NodeDto


class IndependentCascadesModel2(IndependentCascadesModel):
    def __init__(self, number_of_nodes: int = 8, pp: float = 0.5,
                 edges_connections: list = ((0, 1), (1, 2), (2, 4), (2, 6), (4, 3), (6, 7), (6, 5)),
                 seed_indexes: list = [1]):
        super().__init__(number_of_nodes, pp, edges_connections, seed_indexes)

        self.condition_propagation__ = '>'

    def propagate__(self):
        infected = self.infected_nodes_to_visit__.pop(0)
        uninfected_neighbours = self.get_uninfected_neighbours_for_node__(infected)

        if infected not in self.nodes_visited_by_node__:
            self.nodes_visited_by_node__[infected] = set()

        for node in uninfected_neighbours:
            if node in self.nodes_visited_by_node__[infected]:
                continue

            self.nodes_visited_by_node__[infected].add(node)

            infected_neighbours_for_node = self.get_infected_neighbours_for_node__(node)
            infect_probs_by_infected_node = np.random.rand(len(infected_neighbours_for_node))
            infect_prob_for_node = np.sum(infect_probs_by_infected_node) / len(self.get_neighbours_for_node__(node))

            if infect_prob_for_node <= self.pp:
                continue

            self.infected_nodes.append(node)
            self.infected_nodes_views__.append(
                NodeDto(node, is_infected=True, infected_by=infected, infect_prob=infect_prob_for_node))
            self.infected_nodes_to_visit__.append(node)
