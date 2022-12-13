import itertools
import numpy as np

from LinearClassifier import LinearClassifier


class MlpExtreme(LinearClassifier):
    def __init__(self, clf_lin, neurons_hidden_count: int = 100):
        super().__init__()

        self.clf_lin = clf_lin  # można wstawić za klasyfikator liniowy SVM
        self.neurons_hidden_count = neurons_hidden_count

        self.w1 = None
        self.b1 = None

    def fit(self, x: np.array, d: np.array):
        self.w1 = np.zeros((x.shape[1], self.neurons_hidden_count))  # wagi warstry wejściowej, wyznaczyć losując
        self.b1 = np.zeros(self.neurons_hidden_count)  # wagi warstry wejściowej, wyznaczyć losując

        for i in range(self.neurons_hidden_count):
            i1, i2 = np.random.choice(x.shape[0], 2)
            xi, xj = x[i1, :], x[i2, :]

            self.w1[:, i] = np.transpose(xj - xi)
            self.b1[i] = -self.w1[:, i].dot(xi)  # chcemy dodać b do każdej kolumny

        v = self.sigmoid(x.dot(self.w1) + self.b1)
        # v = self.sigmoid(self.w1.T.dot(x) + self.b1)
        self.clf_lin.fit(v, d)  # dowolny klasyfikator liniowy posiadający fit

        self.coefs_ = self.clf_lin.coefs_
        self.intercepts_ = self.clf_lin.intercepts_

    def predict(self, x: np.array):
        v = self.sigmoid(x.dot(self.w1) + self.b1)

        return self.clf_lin.predict(v)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


"""
mnożenie macierzy powinno być łączne względem dodawania
itertools, wylosować sobie wszystkie pary i wybrać
jak mam pkt to jak je odejmę to już mam wektor normalny, a za b trzeba wstawić 1 - wektor normalny

b1 = (xi - xj) * (x - xi) = 0
N * (x - x0) = 0
M.T * x - N.T * x0 = w - b
"""
