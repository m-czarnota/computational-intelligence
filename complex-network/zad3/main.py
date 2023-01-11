import pandas as pd

from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.Model.LinearThresholdModel import LinearThresholdModel
from src.Util.CentralityMeasuresCalculator import CentralityMeasuresCalculator
from src.Factory.Graph.NxGraphFactory import NxGraphFactory


if __name__ == '__main__':
    nx_graph_factory = NxGraphFactory()
    centralityMeasuresCalculator = CentralityMeasuresCalculator()

    edges_data = pd.read_csv('./data/networks/4_1.txt', header=None, delimiter=' ', names=['node1', 'node2', 'weight', 'weight2'])

    nodes_unique1 = set(edges_data['node1'])
    nodes_unique2 = set(edges_data['node2'])
    nodes_unique = nodes_unique1.union(nodes_unique2)

    edges = [tuple(x) for x in edges_data[['node1', 'node2']].to_numpy()]
    weights = edges_data['weight']

    graph = nx_graph_factory.create_nx_graph_from_data(nodes_unique, edges, weights)

    betweenness = centralityMeasuresCalculator.betweenness(graph)
    degree = centralityMeasuresCalculator.degree(graph)

    print(betweenness)
    print(degree)

    # model = IndependentCascadesModel(len(nodes_unique), 0.1, edges, list(betweenness.values())[:5], weights)
    model = LinearThresholdModel(len(nodes_unique), 0.1, edges, list(betweenness.values())[:5], weights)
    # model.animation_interval__ = 200

    model.simulate_propagation(True)
    print(f'% of infected: {len(model.infected_nodes) / model.number_of_node * 100}%')
    print(f'Count of infected: {len(model.infected_nodes)}')

    """
    przygotować funkcje do zapisu
    plik | które centrality | seedy | model | propagation propability | procent zarażonych w sieci 
    można zapisać to w csv
    """
