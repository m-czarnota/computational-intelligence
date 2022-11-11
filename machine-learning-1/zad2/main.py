import pandas as pd
import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split
import time
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from DecisionTree import DecisionTree


def experiment(x, y, model, visualize: bool = True):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.5, test_size=0.5, random_state=0)

    t1 = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting for zoo: {t2 - t1}s')

    t1 = time.time()
    y_pred = model.predict(X_test)
    t2 = time.time()
    print(f'prediction time: {t2 - t1}s')
    print(f'accuracy score: {accuracy_score(y_test, y_pred)}')

    if visualize:
        s = graphviz.Source(model.tree_, format='png')
        s.view()
        # model.tree_.view()


if __name__ == '__main__':
    zoo = pd.read_csv('zoo.csv')

    X = zoo.drop(['animal', 'type'], axis=1)
    X['legs'] = X['legs'] > np.mean(X['legs'])
    Y = zoo['type']

    # experiment(X, Y, DecisionTree())
    print()
    experiment(X, Y, DecisionTreeClassifier())

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
