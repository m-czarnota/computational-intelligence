import pandas as pd
import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split
import time
import graphviz
from sklearn import tree
from DecisionTree import DecisionTree

from Node import Node

if __name__ == '__main__':
    zoo = pd.read_csv('zoo.csv')

    X = zoo.drop(['animal', 'type'], axis=1)
    X['legs'] = X['legs'] > np.mean(X['legs'])
    Y = zoo['type']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.2, test_size=.8)
    decision_tree = DecisionTree()

    t1 = time.time()
    decision_tree.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting for zoo: {t2 - t1}s')

    # rcv1 = fetch_rcv1()
    # X = rcv1['data'] > 0
    # Xr = X[:, 2]
    # Y = rcv1['target'][:, 8]
    # print(X.shape)
    #
    # X = X.tocsc()
    # Y = Y.tocsc()

    # print(X[:, 2])
    # xc = X[X[:, 2].indices > 0]
    # print(xc)

    print('start fitting')
    t1 = time.time()
    # decision_tree.fit(X, Y)
    t2 = time.time()
    print(f'Time of fitting for Reuters: {t2 - t1}s')

    my_tree = '''graph G {
        0 [label="0, clos"]
        0--2 [label="=True"]
        2 [label="2, bank"]
        6 [label="6, dec=1"]
        2--6 [label="=True"]
        5 [label="5, dec=0"]
        2--5 [label="=False"]
        1 [label="1, compan"]
        0--1 [label="=False"]
        4 [label="4, dec=0"]
        1--4 [label="=True"]
        3 [label="3, dec=0"]
        1--3 [label="=False"]
        }'''
    s = graphviz.Source(decision_tree.tree_, format='png')
    s.view()
