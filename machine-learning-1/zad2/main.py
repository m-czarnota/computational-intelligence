import pandas as pd
import numpy as np
from sklearn.datasets import fetch_rcv1
import time
import graphviz
from sklearn import tree
from DecisionTree import DecisionTree
from Metrics import infogain

from Node import Node

if __name__ == '__main__':
    zoo = pd.read_csv('zoo.csv')

    X = zoo.drop(['animal', 'type'], axis=1)
    X['legs'] = X['legs'] > np.mean(X['legs'])
    Y = zoo['type']

    decision_tree = DecisionTree()

    t1 = time.time()
    # decision_tree.fit(X, Y)
    t2 = time.time()
    print(f'Time of fitting for zoo: {t2 - t1}s')

    rcv1 = fetch_rcv1()
    X = rcv1['data'] > 0
    Xr = X[:, 2]
    Y = rcv1['target'][:, 8]
    print(X.shape)

    X = X.tocsc()
    Y = Y.tocsc()

    print(X[:, 2])
    xc = X[X[:, 2].indices > 0]
    print(xc)

    print('start fitting')
    t1 = time.time()
    # decision_tree.fit(X, Y)
    t2 = time.time()
    print(f'Time of fitting for Reuters: {t2 - t1}s')

    # str = tree.export_graphviz(decision_tree)
