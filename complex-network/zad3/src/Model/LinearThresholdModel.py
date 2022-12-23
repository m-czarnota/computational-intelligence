import numpy as np

from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.DTO.NodeDto import NodeDto


class LinearThresholdModel(IndependentCascadesModel):
    def __init__(self, number_of_nodes: int = 8, pp: float = 0.5,
                 edges_connections: list = ((0, 1), (1, 2), (2, 4), (2, 6), (4, 3), (6, 7), (6, 5)),
                 seed_indexes: list = [1],
                 weights: list = None):
        super().__init__(number_of_nodes, pp, edges_connections, seed_indexes)

        self.weights = weights

        self.default_filename__ = 'linear_threshold_model'
        self.condition_propagation__ = '>'

    def propagate__(self):
        recently_infected = self.infected_nodes_to_visit__.pop(0)
        newly_infected = []

        for infected in recently_infected:
            uninfected_neighbours = self.get_uninfected_neighbours_for_node__(infected)

            for node in uninfected_neighbours:
                infected_neighbours_for_node = self.get_infected_neighbours_for_node__(node)
                infect_probs_by_infected_node = np.random.rand(len(infected_neighbours_for_node)) if self.weights is None else self.get_weights_edges_for_node_and_nodes__(node, infected_neighbours_for_node)
                infect_prob_for_node = np.sum(infect_probs_by_infected_node) / len(self.get_neighbours_for_node__(node))

                if infect_prob_for_node <= self.pp:
                    continue

                self.infected_nodes.append(node)
                self.infected_nodes_views__.append(
                    NodeDto(node, is_infected=True, infected_by=infected, infect_prob=infect_prob_for_node))
                newly_infected.append(node)

        if len(newly_infected):
            self.infected_nodes_to_visit__.append(newly_infected)

    def get_weights_edges_for_node_and_nodes__(self, node: int, nodes: list):
        return list(map(lambda x: self.get_weight_edge_for_2_nodes__(x, node), nodes))
