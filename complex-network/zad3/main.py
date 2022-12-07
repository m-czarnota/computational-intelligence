import networkx as nx

from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.Model.IndependentCascadesModel2 import IndependentCascadesModel2

if __name__ == '__main__':
    graph = nx.barabasi_albert_graph(16, 2)

    model = IndependentCascadesModel2(len(graph.nodes), 0.21, [edge for edge in graph.edges], [1, 15])
    model.animation_interval__ = 1000

    model.save_animation_to_file()
