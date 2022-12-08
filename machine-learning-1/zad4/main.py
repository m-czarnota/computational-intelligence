import time
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from AveragedPerceptron import AveragedPerceptron
from MlpExtreme import MlpExtreme
from Svm1 import Svm1
from Svm2 import Svm2
from Svm2Sparse import Svm2Sparse
from VotedPerceptron import VotedPerceptron
from LinearClassifier import LinearClassifier
from Perceptron import Perceptron
from MlpTest import MlpTest


def linear_separable_dataset():
    w, b = [1, 1], -1
    x = np.random.randn(100, 2)
    d = np.sign(x.dot(w) + b)

    return x, d


def non_linear_separable_dataset():
    w, b = [1, 1], -1
    x = np.random.randn(100, 2)

    d = np.random.rand(x.shape[0])
    d[d < 0.5] = -1
    d[d >= 0.5] = 1

    return x, d


def chessboard_dataset(n: int = 1000, m: int = 3):
    X = np.random.rand(n, 2) * 2 - 1
    y = np.zeros(n)
    X = np.random.rand(n, 2) * m
    y = np.mod(np.sum(np.floor(X), axis=1), 2) * 2. - 1.
    X = X + np.random.randn(*X.shape) * 0.1

    return X, y


def plot_class(x: np.array, y: np.array, clf: LinearClassifier):
    n, m = x.shape

    [x1, x2] = np.meshgrid(x[:, 0], x[:, 1])
    x_1_2_flatten = np.array([x1.flatten(), x2.flatten()]).T
    print(x_1_2_flatten)

    z = clf.predict(x_1_2_flatten)
    z = z.reshape((n, n))

    plt.contourf(x1, x2, z)
    plt.show()


def normalize_decisions(d):
    d_normalized = np.ones(d.shape[0]).astype("int8")
    d_normalized[d == np.unique(d)[0]] = -1

    return d_normalized


def experiment(x, d):
    perceptron = VotedPerceptron()

    t1 = time.time()
    w, b = perceptron.fit(x, d)
    t2 = time.time()
    print(f'Time of fitting: {t2 - t1}s.\nNumber of iterations: {perceptron.iteration_count}')

    perceptron.plot_class(x, d)

    # plt.figure()
    # plt.scatter(x[:, 0], x[:, 1], c=d)
    #
    # x1 = np.array([np.min(x[:, 0]), np.max(x[:, 1])])
    # x2 = -(b + w[0] * x1) / w[1]
    # plt.plot(x1, x2)
    #
    # plt.show()


def svm_test():
    """
    porównać 2 warianty svm
    czy to rozwiązanie które mamy jest efektywne?
    zwrócić uwagę na macierz G - są tam macierze rzadkie, w macierzy p też
    jak zwiększymy liczbę próbek to macierz się rozszerzy i będzie w 1/4 wypełniona
    może się zdarzyć, że x będzie macierzą rzadką. więc lepiej byłoby trzymać macierz g jako macierz rzadką
    [[]mxn  []mxm(jednostkowa)]
     []mxm(rzadka)  []mxm(jednostkowa)]
    cvx.matrix zamienić na cvx.sv_matrix
    wszystkie te macierze mogą być rzadkie, powinny być rzadkie macierze g i p
    można zrobić svm2 sparse i sprawdzić czy to pomoże
    nie będzie dużego zysku jeżeli macierz x jest gęsta i n jest duże
    będzie zysk jak n jest małe w stosunku do m
    """

    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([-1, -1, -1, 1])

    n = 50
    X = np.vstack((np.random.randn(n, 2), np.random.randn(n, 2) + 2))
    y = np.concatenate((np.ones(n), -np.ones(n)))

    clf = Svm2(c=1e-3)
    clf.fit(X, y)
    clf.plot_class(X, y, True)


def mlp_scikit_learn_test():
    X, y = datasets.make_circles(1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    mlp = MLPClassifier((50, 40), max_iter=1000, alpha=0.01)

    t1 = time.time()
    mlp.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting: {t2 - t1}s')

    LinearClassifier.plot_class_universal(mlp, X_test, y_test)


def mlp_extreme_test():
    X, y = MlpTest.generate_chessboard_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    svm2 = Svm2()
    mlp = MlpExtreme(svm2, 100)

    t1 = time.time()
    mlp.fit(X, y)
    t2 = time.time()
    print(f'Time of fitting: {t2 - t1}s')

    print()
    mlp.plot_class(X, y)


if __name__ == '__main__':
    # svm_test()
    # mlpTest = MlpTest()
    # mlpTest.experiment()
    mlp_extreme_test()

    # x_data, decisions = linear_separable_dataset()
    # experiment(x_data, decisions)

    # x_data, decisions = non_linear_separable_dataset()
    # experiment(x_data, decisions)
    #
    # sonar_data = pd.read_csv('sonar_csv.csv')
    # decisions = sonar_data[sonar_data.columns[-1]]
    # decisions = normalize_decisions(decisions)
    # x_data = sonar_data.drop(sonar_data.columns[-1], axis=1).to_numpy()
    #
    # experiment(x_data, decisions)
