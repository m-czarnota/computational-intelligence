from typing import Tuple

import pandas as pd


class NetworkFileReader:
    @staticmethod
    def read_properties(filename: str) -> Tuple:
        edges_data = pd.read_csv(filename, header=None, delimiter=' ',
                                 names=['node1', 'node2', 'weight', 'weight2'])

        nodes_unique1 = set(edges_data['node1'])
        nodes_unique2 = set(edges_data['node2'])
        nodes_unique = nodes_unique1.union(nodes_unique2)

        edges = [tuple(x) for x in edges_data[['node1', 'node2']].to_numpy()]
        weights = edges_data['weight']

        return nodes_unique, edges, weights
