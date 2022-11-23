import networkx as nx
import numpy as np

from src.DTO.NodeDto import NodeDto


class ProgModel:
    def __init__(self, number_of_nodes: int = 8,
                 edges_connections: list = ((0, 1), (1, 2), (2, 4), (2, 6), (4, 3), (6, 7), (6, 5)),
                 seed_indexes: list = [1]):
        self.graph = None
        self.pp = 0.5

        self.number_of_node = number_of_nodes
        self.edges_connections = edges_connections

        self.infected_nodes = seed_indexes
        self.infected_nodes_probs = {index: 1 for index in seed_indexes}
        self.seeds = seed_indexes
        self.nodes_visited_by_node = {}

        self.initialise_graph__()

    def initialise_graph__(self):
        self.graph = nx.Graph()
        [self.graph.add_node(i) for i in range(self.number_of_node)]
        [self.graph.add_edge(edge[0], edge[1]) for edge in self.edges_connections]

    def propagate(self):
        visited_by_node = {}
        infected_nodes_to_visit = self.seeds.copy()

        while len(infected_nodes_to_visit) > 0:
            infected = infected_nodes_to_visit.pop(0)
            nodes_to_visit = self.get_nodes_to_visit_for_node__(infected)

            if infected not in self.nodes_visited_by_node:
                self.nodes_visited_by_node[infected] = set()

            for node in nodes_to_visit:
                if node in self.nodes_visited_by_node[infected]:
                    continue

                self.nodes_visited_by_node[infected].add(node)
                infect_prob = np.random.rand()
                if infect_prob >= self.pp:
                    continue

                self.infected_nodes.append(node)
                self.infected_nodes_probs[node] = infect_prob
                infected_nodes_to_visit.append(node)

        print(self.infected_nodes)
        print(self.infected_nodes_probs)

        # while True:
        #     for node_index in self.infected_indexes:
        #         list(filter(lambda x: node_index in x, self.graph.edges))
        #         edges_list = list(filter(lambda x: node_index in x, self.graph.edges))
        #         edges_to_visit = list(map(lambda edges: list(filter(lambda edge_val: edge_val != node_index, edges))[0], edges_list))
        #
        #         print(edges_list, edges_to_visit)
        #
        #         for edge in edges_to_visit:
        #             if node_index not in self.visited.keys():
        #                 self.visited[node_index] = []
        #
        #             if edge == node_index:
        #                 continue
        #
        #             self.visited[node_index].append(edge)
        #             infect_prob = np.random.rand()
        #
        #             if infect_prob < self.pp:
        #                 self.infected_indexes.add(edge)
        #
        #     break

    def get_nodes_to_visit_for_node__(self, node: int):
        edges_list = list(filter(lambda x: node in x, self.graph.edges))
        node_edges = list(
            map(lambda edges: list(filter(lambda edge_val: edge_val != node, edges))[0], edges_list))

        nodes_to_visit = [node_edge for node_edge in node_edges if node_edge not in self.infected_nodes]

        return nodes_to_visit
