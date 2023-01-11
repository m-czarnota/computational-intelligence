import networkx as nx


class CentralityMeasuresCalculator:
    def betweenness(self, g: nx.Graph) -> dict:
        centrality = nx.betweenness_centrality(g)

        return self.__sort_results(centrality)

    def degree(self, g: nx.Graph) -> dict:
        centrality = nx.degree_centrality(g)

        return self.__sort_results(centrality)

    @staticmethod
    def __sort_results(results: dict, reverse: bool = True) -> dict:
        return dict(sorted(results.items(), key=lambda item: item[1], reverse=reverse))
