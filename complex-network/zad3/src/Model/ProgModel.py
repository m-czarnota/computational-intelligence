import networkx as nx
import numpy as np


class ProgModel:
    def __init__(self, number_of_nodes: int = 8, edges_connections: list = ((0, 1), (1, 2), (2, 4), (2, 6), (4, 3), (6, 7), (6, 5))):
        self.graph = None
        self.pp = 0.5
        self.number_of_node = number_of_nodes
        self.edges_connections = edges_connections
        self.seed_index = 1
        self.infected_indexes = {self.seed_index}
        self.visited = {}

        self.initialise_graph()

    def initialise_graph(self):
        self.graph = nx.Graph()
        [self.graph.add_node(i) for i in range(self.number_of_node)]
        [self.graph.add_edge(edge[0], edge[1]) for edge in self.edges_connections]

    def propagate(self):
        while True:
            for node_index in self.infected_indexes:
                list(filter(lambda x: node_index in x, self.graph.edges))
                edges = list(filter(lambda x: node_index in x, self.graph.edges))

                print(edges)

                break

                for edge in edges:
                    if node_index not in self.visited.keys():
                        self.visited[node_index] = []

                    if edge == node_index:
                        continue

                    self.visited[node_index].append(edge)
                    infect_prob = np.random.rand()

                    if infect_prob < self.pp:
                        self.infected_indexes.add(edge)

            break

        print(self.visited)
        print(self.infected_indexes)
