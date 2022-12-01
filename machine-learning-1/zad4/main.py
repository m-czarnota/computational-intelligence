# xi, di
# xi - współrzędne x
# di - decyzje
# <w, x> to wektor
# <w, x> + b
#
# def perceptron(x, d):
#     n == lr
#     stop_condition = n < count_samples
#     while (not stop_condition):
#         mozna dodać warunek bezpieczeństwa na czas
#
#         iterowanie po próbkach i
#
#         if (d[i] * iloczyn_skalarny(w, x) + b <= 0):
#             #update
#             w = w + d[i] * x[i]
#             b = b + d[i] * 1
#             n = 0
#         else:
#             n += 1
#
#         return w, b
#
# są próbki dodatnie, są próbki ujemne. chcemy znaleźć prostą która separuje próbki
import time

from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

from AveragedPerceptron import AveragedPerceptron
from Svm2 import Svm2
from VotedPerceptron import VotedPerceptron

# def dane_liniowo_separowalne():
#     w, b = [1, 3], -1
#     x = np.random.rand(10, 2) * 3
#     d = np.sign(x.dot(w) + b)
#     x.dot(w) - iloczyn skalarny <w, x>
#
# w i b to parametry prostej (hiperpłaszczyzny - równanie hiperpłaszczyzny: <w, x> + b = 0, czyli sum(w[i] * x[i] + b) = 0)
# jak znaleźć b? mając pkt (0,0) liczymy x/||w||
#
# aby narysować prostą, trzeba wyznaczyć dwa pkt

# def funkcja_uniwersalna_brzydka():
#     contour
#     [x1, x2] = meshgrid(zakres)  # daje nam 2 macierze współrzędnych
#     X = np.array([x1.flatten(), x2.flatten()]).T  # flatten robi kopię, raven modyfikuje oryginalne
#
#     z = clf.predict(X)
#     z = z.reshape((n, n))
#     contour(x1, x2, z, [0, 0])

"""
averaged uśredniamy sumę wag, dla tych która była dobra
uśredniamy wagi, lista w, b. ile razy wagi były dobre i uśredniamy

do margin dodajemy distance = True. jak jest 
jeset prosta. to odległość pkt od prostej to d=<w, x> + b / ||w|| -> odległość
margines (<w, x> + b) * di**2
jak distance == True to dzielimy przez ||w||, jak False to nie dzielimy

trzeba pomnożyć kolumna * kolumna, więc element przez element
w matlabie jest mnożenie .*
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LinearClassifier import LinearClassifier
from Perceptron import Perceptron


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


def plot_class_universal(clf, x, y, is_line: bool = False):
    x1_min = np.min(x[:, 0]) - 0.5
    x1_max = np.max(x[:, 0]) + 0.5

    plt.figure()

    if is_line:
        points = np.array([[i, -(clf.coefs_[0] * i + clf.intercepts_) / clf.coefs_[1]] for i in np.linspace(x1_min, x1_max)])
        plt.plot(points[:, 0], points[:, 1], 'k')
    else:
        x2_min = np.min(x[:, 1]) - 0.5
        x2_max = np.max(x[:, 1]) + 0.5

        number_of_points = 250
        xn, yn = np.meshgrid(np.linspace(x1_min, x1_max, number_of_points), np.linspace(x2_min, x2_max, number_of_points))
        zn = clf.predict(np.c_[xn.flatten(), yn.flatten()]).reshape(xn.shape)

        plt.contourf(xn, yn, zn, cmap=ListedColormap(['y', 'r']))

    # !!circular import error of Linear Classifier!!
    # if isinstance(self, Svm1):
    #     indexes = self.svinds_
    #     plt.plot(x[indexes, 0], x[indexes, 1], 'co', markersize=10, alpha=0.5)

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=ListedColormap(['b', 'g']))
    plt.title('Boundary of separation')

    plt.show()


def mlp_scikit_learn_test():
    X, y = chessboard_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    mlp = MLPClassifier((50, 40), max_iter=1000, alpha=0.01)

    t1 = time.time()
    mlp.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting: {t2 - t1}s')

    plot_class_universal(mlp, X_test, y_test)


if __name__ == '__main__':
    # svm_test()
    mlp_scikit_learn_test()

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
