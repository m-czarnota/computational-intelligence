import networkx as nx
from matplotlib import pyplot as plt

from src.Model.ProgModel import ProgModel

if __name__ == '__main__':
    progModel = ProgModel()
    progModel.propagate()

    nx.draw(progModel.graph, with_labels=True)
    plt.show()
