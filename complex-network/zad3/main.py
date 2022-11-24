import networkx as nx

from src.Model.IndependentCascadesModel import IndependentCascadesModel

if __name__ == '__main__':
    graph = nx.watts_strogatz_graph(16, 3, 0.3)

    model = IndependentCascadesModel(len(graph.nodes), 0.5, [edge for edge in graph.edges], [1, 15])
    model.animation_interval__ = 1000

    model.save_animation_to_file()
