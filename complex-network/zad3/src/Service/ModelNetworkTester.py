import networkx as nx
import numpy as np
import pandas as pd
import pickle
from os.path import exists

from src.Factory.Graph.NxGraphFactory import NxGraphFactory
from src.Model.IndependentCascadesModel import IndependentCascadesModel
from src.Model.LinearThresholdModel import LinearThresholdModel
from src.Service.NetworkFileReader import NetworkFileReader
from src.Util.CentralityMeasuresCalculator import CentralityMeasuresCalculator
from src.Util.VerbosityHelper import VerbosityHelper


class ModelNetworkTester:
    def __init__(self, verbosity_level: int = 0):
        self._verbosityHelper = VerbosityHelper(bool(verbosity_level), verbosity_level)
        self._nxGraphFactory = NxGraphFactory()
        self._centralityMeasuresCalculator = CentralityMeasuresCalculator()
        self._networkFileReader = NetworkFileReader()

        self.results_filepath = './data'
        centrality_measures_pickle_name = 'centrality_measures.p'

        self._path_to_pickled_centrality_measures = f'{self.results_filepath}/pickled/{centrality_measures_pickle_name}'
        self._centrality_measures: dict = pickle.load(open(self._path_to_pickled_centrality_measures, 'rb')) if exists(self._path_to_pickled_centrality_measures) else {}

        # self._propagation_probabilities = np.arange(20, 100, 20) / 100
        self._propagation_probabilities = [0.2]
        # self._seed_fraction = np.arange(5, 20, 5) / 100
        self._seed_fraction = [0.05]

    def test_for_networks(self, filenames: list):
        results = pd.DataFrame()

        for filename in filenames:
            for network_results in self.test_for_network(filename):
                results = pd.concat([results, network_results.to_frame().T])

        results.to_csv(f'{self.results_filepath}/results.csv')

    def test_for_network(self, filename: str) -> list:
        network_results = []

        nodes_unique, edges, weights = self._networkFileReader.read_properties(filename)
        nodes_count = len(nodes_unique)

        graph = self._nxGraphFactory.create_nx_graph_from_data(nodes_unique, edges, weights)
        centrality_measures = self._calc_centrality_measures_for_network(graph, filename)

        for degree_name, degree_result in centrality_measures.items():
            degree_result = list(degree_result)

            for pp in self._propagation_probabilities:
                for seed_fraction_percent in self._seed_fraction:
                    seeds_count = np.round(nodes_count * seed_fraction_percent).astype('int')
                    degree_results_for_test = degree_result[:seeds_count]

                    independent = IndependentCascadesModel(nodes_count, pp, edges, degree_results_for_test, weights)
                    independent.simulate_propagation()

                    linear = LinearThresholdModel(nodes_count, pp, edges, degree_results_for_test, weights)
                    linear.simulate_propagation()

                    for model in [independent]:
                        result = {
                            'N': filename,
                            'PP': pp,
                            'SM': degree_name,
                            'SF': seeds_count,
                            'SF%': f'{seed_fraction_percent}%',
                            'M': model.__class__.__name__,
                            'IF%': f'{(len(model.infected_nodes) / model.number_of_node * 100):.4f}%',
                        }

                        if self._verbosityHelper.verbosity_level:
                            result = {**result, **{
                                'Propagate history': model.propagate_history,
                                'IF': model.infected_nodes,
                                'Seeds': degree_result,
                            }}

                        network_results.append(pd.Series(result))

        return network_results

    def _calc_centrality_measures_for_network(self, graph: nx.Graph, network_name: str) -> dict:
        if network_name in self._centrality_measures.keys():
            return self._centrality_measures[network_name]

        betweenness = self._verbosityHelper.verbose(self._centralityMeasuresCalculator.betweenness, [graph], 2, 'Time of calculating betweenness centrality')
        degree = self._verbosityHelper.verbose(self._centralityMeasuresCalculator.degree, [graph], 2, 'Time of calculating degree centrality')

        self._centrality_measures[network_name] = {
            'betweenness': betweenness,
            'degree': degree,
        }

        nodes_count = len(graph.nodes)
        random_count_seeds = np.random.randint(nodes_count)
        self._centrality_measures[network_name]['random'] = np.random.randint(nodes_count, size=random_count_seeds)

        pickle.dump(self._centrality_measures, open(self._path_to_pickled_centrality_measures, 'wb'))

        return self._centrality_measures[network_name]
