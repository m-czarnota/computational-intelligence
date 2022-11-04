import pandas as pd
import numpy as np
import time
from DecisionTree import DecisionTree
import graphviz
from sklearn import tree

from Node import Node

if __name__ == '__main__':
    zoo = pd.read_csv('zoo.csv')

    X = zoo.drop(['animal', 'type'], axis=1)
    X['legs'] = X['legs'] > np.mean(X['legs'])
    Y = zoo['type']
    # info_gains = {key: entropy(autos[key]) for key in autos.columns}
    # print(sorted(info_gains.items(), key=lambda x: x[1], reverse=True))

    decision_tree = DecisionTree()
    decision_tree.fit(X, Y)

    # decision_tree.root = Node()
    # decision_tree.fit(X, Y)
    # decision_tree.root.column = 'legs'
    # decision_tree.root.impurity_value = 4.532
    #
    # left = Node()
    # left.parent = decision_tree.root
    # left.column = 'catsize'
    # left.impurity_value = 5.432
    # decision_tree.root.left = left
    #
    # right = Node()
    # right.parent = decision_tree.root
    # right.column = 'fins'
    # right.impurity_value = 3.432
    # decision_tree.root.right = right
    #
    # str = tree.export_graphviz(decision_tree)
