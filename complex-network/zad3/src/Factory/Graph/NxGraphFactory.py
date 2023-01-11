import networkx as nx


class NxGraphFactory:
    @staticmethod
    def create_nx_graph_from_data(nodes: iter, edges: iter, weights: iter) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        [graph.add_edge(*edge, weight=weights[edge_iter]) for edge_iter, edge in enumerate(edges)]

        return graph
