import time
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AveragedPerceptron import AveragedPerceptron
from MlpBackPropagation import MlpBackPropagation
from MlpExtreme import MlpExtreme
from Svm1 import Svm1
from Svm2 import Svm2
from SvmTest import SvmTest
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
    clf.plot_class(X, y, True, clf.sv_indexes_)


def mlp_scikit_learn_test():
    X, y = datasets.make_circles(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    mlp = MLPClassifier(250, max_iter=1000, alpha=0.01)

    t1 = time.time()
    mlp.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting: {t2 - t1}s')

    LinearClassifier.plot_class_universal(mlp, X_test, y_test)


def mlp_extreme_test():
    X, y = datasets.make_circles(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    svm2 = LogisticRegression()
    mlp = MlpExtreme(svm2, 250)

    t1 = time.time()
    mlp.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting: {t2 - t1}s')

    print()
    LinearClassifier.plot_class_universal(mlp, X_test, y_test)


def mlp_back_prop_test():
    X, y = MlpTest.generate_spirals_dataset(1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    mlp = MlpBackPropagation(neurons_hidden_count=250)

    t1 = time.time()
    mlp.fit(X_train, y_train)
    t2 = time.time()
    print(f'Time of fitting {t2 - t1}s')

    LinearClassifier.plot_class_universal(mlp, X_test, y_test)


if __name__ == '__main__':
    # svm_test()
    # mlp_scikit_learn_test()
    # mlp_extreme_test()
    # mlp_back_prop_test()

    # svm_test = SvmTest()
    # svm_test.experiment()

    mlpTest = MlpTest()
    mlpTest.experiment()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(mlpTest.data_table.to_markdown())


"""
zewnętrzna pętla po alhpa
train_and_split
pętla po klasyfikatorach
lepiej w ten sposób, bo można porównywać modele między sobą dla takich samych zbiorów

np.logspace
jak alhpa = 10**-2, to biorę tę część do uczenia, a całą resztę do testowania
wnioski do złożoności próbkowej do wykresu!
grid_search który znajdzie parametr C - powinien on być na części uczącej

zastanowić się, czy można dane zwizualizować
zrobić PCA i zwizualizować
dla MNIST uda się bez problemu
co dla bazy Reuters to trzeba się zastanowić
eig dla macierzy gęstej będzie bardzo długo się liczyć
dla Reuters macierz kowariancji będzie miała 40000x40000 rozmiar

zapisywać do tabelki pandas i zrobić groupby, aby zaregować
i w ten sposób przedstawić

parametr alpha, co oznacza, że klasyfikator jest regularyzowany
można pominąć niektóre klasyfikatory, jak voted i averaged perceptron
tabelka z podsumowaniem na koniec
DataFrame.agg - przyjmuje słownik wartości {'kolumna': 'funkcja'}, np. {'AUC': 'mean'}
zamiast mean można dać funkcję agregującą, która da +-, aby wypisać 97.54% +- 0.05%
dla każdej miary można osobną tabelkę przedstawić
w pandas jest pivot table

regresja logistyczna przynajmniej 3: L1, L2 i elastic
grid_search!
można wziąć wbudowany perceptron, bo może być szybszy niż nasz, chociaż nie trzeba, bo długo się liczyć będzie
SVC (linear)
SVM (nasza implementacja)
MLPClassifier (bez hidden layers, będzie wtedy liniowy)
LogisticRegression (l1, l2, elementalist)
"""