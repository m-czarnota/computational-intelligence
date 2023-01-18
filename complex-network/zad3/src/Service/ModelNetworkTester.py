import networkx as nx
import numpy as np
import pandas as pd

from src.Factory.Graph.NxGraphFactory import NxGraphFactory
from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.Model.LinearThresholdModel import LinearThresholdModel
from src.Service.NetworkFileReader import NetworkFileReader
from src.Util.CentralityMeasuresCalculator import CentralityMeasuresCalculator


class ModelNetworkTester:
    def __init__(self):
        self.nxGraphFactory = NxGraphFactory()
        self.centralityMeasuresCalculator = CentralityMeasuresCalculator()
        self.networkFileReader = NetworkFileReader()

        self._network_degrees = {}
        # self._propagation_probabilities = np.arange(20, 100, 20) / 100
        self._propagation_probabilities = [0.2]
        # self._seed_fraction = np.arange(5, 20, 5) / 100
        self._seed_fraction = [0.05]

    def test_for_networks(self, filenames: list, verbose: bool = False):
        results = pd.DataFrame()

        for filename in filenames:
            for network_results in self.test_for_network(filename, verbose):
                results = pd.concat([results, network_results.to_frame().T])

        print(results.to_markdown())

    def test_for_network(self, filename: str, verbose: bool = False) -> list:
        network_results = []

        nodes_unique, edges, weights = self.networkFileReader.read_properties(filename)
        nodes_count = len(nodes_unique)

        graph = self.nxGraphFactory.create_nx_graph_from_data(nodes_unique, edges, weights)
        network_degrees = self._calc_degrees_for_network(graph, filename)

        for degree_name, degree_result in network_degrees.items():
            degree_result = list(degree_result)

            for pp in self._propagation_probabilities:
                for seed_fraction_percent in self._seed_fraction:
                    seeds_count = np.round(nodes_count * seed_fraction_percent).astype('int')
                    degree_results_for_test = degree_result[:seeds_count]

                    independent = IndependentCascadesModel(nodes_count, pp, edges, degree_results_for_test, weights)
                    independent.simulate_propagation()

                    # linear = LinearThresholdModel(nodes_count, pp, edges, degree_results_for_test, weights)
                    # linear.simulate_propagation()

                    for model in [independent]:
                        result = pd.Series({
                            'N': filename,
                            'PP': pp,
                            'SM': degree_name,
                            'SF': seeds_count,
                            'SF%': f'{seed_fraction_percent}%',
                            'M': model.__class__.__name__,
                            'IF': model.infected_nodes,
                            'IF%': f'{len(model.infected_nodes) / model.number_of_node * 100}%'
                        })
                        network_results.append(result)

        return network_results

    def _calc_degrees_for_network(self, graph: nx.Graph, network_name: str) -> dict:
        if network_name in self._network_degrees.keys():
            return self._network_degrees[network_name]

        self._network_degrees[network_name] = {
            'betweenness': self.centralityMeasuresCalculator.betweenness(graph),
            'degree': self.centralityMeasuresCalculator.degree(graph),
        }

        nodes_count = len(graph.nodes)
        random_count_seeds = np.random.randint(nodes_count)
        self._network_degrees[network_name]['random'] = np.random.randint(nodes_count, size=random_count_seeds)

        return self._network_degrees[network_name]
