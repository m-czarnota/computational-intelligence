import networkx as nx
import pandas as pd

from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.Model.LinearThresholdModel import LinearThresholdModel

if __name__ == '__main__':
    edges_data = pd.read_csv('./data/networks/12_1.txt', header=None, delimiter=' ', names=['node1', 'node2', 'weight', 'weight2'])

    nodes_unique1 = set(edges_data['node1'])
    nodes_unique2 = set(edges_data['node2'])
    nodes_unique = nodes_unique1.union(nodes_unique2)

    edges = [tuple(x) for x in edges_data[['node1', 'node2']].to_numpy()]
    weights = edges_data['weight']

    # model = IndependentCascadesModel(len(nodes_unique), 0.1, edges, [0], weights)
    model = LinearThresholdModel(len(nodes_unique), 0.1, edges, [0], weights)
    # model.animation_interval__ = 200

    model.simulate_propagation(True)
    print(f'% of infected: {len(model.infected_nodes) / model.number_of_node}%')
    print(f'Count of infected: {len(model.infected_nodes)}')
