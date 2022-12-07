from time import sleep
import networkx as nx
import numpy as np

from src.Animator.NxGraphAnimator import NxGraphAnimator
from src.DTO.NodeDto import NodeDto
from src.Enum.NodeColorEnum import NodeColorEnum


class IndependentCascadesModel(NxGraphAnimator):
    def __init__(self, number_of_nodes: int = 8, pp: float = 0.5,
                 edges_connections: list = ((0, 1), (1, 2), (2, 4), (2, 6), (4, 3), (6, 7), (6, 5)),
                 seed_indexes: list = [1]):
        super().__init__()

        self.pp = pp

        self.number_of_node = number_of_nodes
        self.edges_connections = edges_connections
        self.node_colors = []

        self.seeds = seed_indexes
        self.infected_nodes = seed_indexes
        self.nodes_visited_by_node__ = {}

        self.infected_nodes_views__ = []
        self.infected_nodes_to_visit__ = []

        self.default_filename__ = 'independent_cascades_model'
        self.condition_propagation__ = '<'

        self.initialise_new_random_graph__()

    def initialise_new_random_graph__(self):
        self.graph = nx.Graph()
        [self.graph.add_node(i) for i in np.arange(self.number_of_node)]
        [self.graph.add_edge(edge[0], edge[1]) for edge in self.edges_connections]

    def prepare_to_draw__(self):
        self.prepare_to_propagate__()

        self.node_colors = np.array([NodeColorEnum.NO_INFECTED.value for node in self.graph.nodes])
        self.node_colors[self.seeds] = NodeColorEnum.SEED.value

        super().prepare_to_draw__()

    def simulate_propagation(self):
        self.prepare_to_propagate__()

        while len(self.infected_nodes_to_visit__) > 0:
            self.propagate__()

    def get_neighbours_for_node__(self, node: int):
        edges_list = list(filter(lambda x: node in x, self.graph.edges))
        node_edges = list(map(lambda edges: list(filter(lambda edge_val: edge_val != node, edges))[0], edges_list))

        return node_edges

    def get_uninfected_neighbours_for_node__(self, node: int):
        node_edges = self.get_neighbours_for_node__(node)
        nodes_to_visit = [node_edge for node_edge in node_edges if node_edge not in self.infected_nodes]

        return nodes_to_visit

    def get_infected_neighbours_for_node__(self, node: int):
        node_edges = self.get_neighbours_for_node__(node)
        nodes_to_visit = [node_edge for node_edge in node_edges if node_edge in self.infected_nodes]

        return nodes_to_visit

    def propagate__(self):
        infected = self.infected_nodes_to_visit__.pop(0)
        nodes_to_visit = self.get_uninfected_neighbours_for_node__(infected)

        if infected not in self.nodes_visited_by_node__:
            self.nodes_visited_by_node__[infected] = set()

        for node in nodes_to_visit:
            if node in self.nodes_visited_by_node__[infected]:
                continue

            self.nodes_visited_by_node__[infected].add(node)
            infect_prob = np.random.rand()
            if infect_prob >= self.pp:
                continue

            self.infected_nodes.append(node)
            self.infected_nodes_views__.append(
                NodeDto(node, is_infected=True, infected_by=infected, infect_prob=infect_prob))
            self.infected_nodes_to_visit__.append(node)

    def prepare_to_propagate__(self):
        self.infected_nodes = self.seeds.copy()
        self.nodes_visited_by_node__ = {}

        self.infected_nodes_views__ = []
        self.infected_nodes_to_visit__ = self.seeds.copy()

    def update_animation__(self, frame):
        if len(self.infected_nodes_to_visit__) == 0:
            return

        if self.actual_node == 0:
            sleep(0.5)

        self.propagate__()
        self.draw_node__()
        self.draw_edge__()

    def draw_node__(self):
        self.actual_node += 1

        self.node_colors[self.infected_nodes] = NodeColorEnum.INFECTED.value
        self.node_colors[self.seeds] = NodeColorEnum.SEED.value

        nx.draw(self.graph, with_labels=True, pos=self.pos, node_color=self.node_colors)

    def draw_edge__(self):
        weight_key = 'weight'
        self.actual_edge += 1

        for node_view in self.infected_nodes_views__:
            edge_to_check = (node_view.infected_by, node_view.index)
            edge_to_check_reversed = tuple(reversed(edge_to_check))

            edge = list(filter(lambda x: x == edge_to_check or x == edge_to_check_reversed, self.graph.edges))[0]
            self.graph.edges[edge][weight_key] = f'{node_view.infect_prob:.2} {self.condition_propagation__} {self.pp}'

        edges_list = {edge: self.graph.edges[edge][weight_key] if weight_key in self.graph.edges[edge] else ''
                      for edge in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, pos=self.pos, edge_labels=edges_list)
