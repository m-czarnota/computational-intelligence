"""
dla każdego k powtórzyć eksperyment kilka razy i usrednić, żeby wykres nie był poszarpany
na wykresie zaznaczamy optymalną wartosć parametru k
powinno być conajmniej 1000 pkt

dzielimy zbiór na pół. y_test to wyniki dla idealnej
sprawdzać od k=1 do k=rozmiar_proby, jak rozmiar_proby == 500, to 500
dla małych k dać krok mały, dla większych dać większy - do 20 co 2, do 50 co 5 na przykład
"""


import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

CHESSBOARD_SAMPLE_COUNT = 1000


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_chessboard(n: int, gauss: bool = False):
    second_dim = math.sqrt(n)

    x = np.random.rand(CHESSBOARD_SAMPLE_COUNT, 2)

    y = np.add(((np.floor(x[:, 0] * second_dim)) % second_dim), ((np.floor(x[:, 1] * second_dim)) % second_dim))
    y = y % 2

    if gauss is True:
        std = get_truncated_normal(low=0, upp=0.5)
        x[:, 0] += std.rvs()
        x[:, 1] += std.rvs()

    return x, y


def zad2(X, Y):
    k_dict = {}
    k = 1
    k_step = 10
    max_score = 0
    k_optimal = 0

    for i in range(1, CHESSBOARD_SAMPLE_COUNT):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)

        new_score = neigh.score(X_test, y_test)
        if k in k_dict.keys():
            k_dict[k].append(new_score)
        else:
            k_dict[k] = [new_score]

        if new_score > max_score:
            max_score = new_score
            k_optimal = k

        if i % k_step == 0:
            k += 1

    print(max_score, k_optimal)
    for key, values in k_dict.items():
        k_dict[key] = np.mean(values)

    plt.figure()
    plt.scatter(k_dict.keys(), k_dict.values())
    plt.title(f'score by k; k is avg for different data splits, k_step={k_step}; the best k={k_optimal}, max_score={max_score}')
    plt.xlabel('k')
    plt.ylabel('score')
    plt.grid()
    plt.show()


def zad3(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    k = 6
    algorithms = ['brute', 'kd_tree', 'ball_tree']

    times_of_fit = {}
    times_of_score = {}
    scores = []

    for algorithm in algorithms:
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm)

        time1 = time.time()
        neigh.fit(X_train, y_train)
        time2 = time.time()
        times_of_fit[algorithm] = time2 - time1

        time1 = time.time()
        score = neigh.score(X_test, y_test)
        time2 = time.time()
        times_of_score[algorithm] = time2 - time1
        scores.append(score)

    plt.figure()
    plt.title(f'differences in time of fit by algorithm, k={k}')
    plt.bar(times_of_fit.keys(), times_of_score.values(), label='time of fit by algorithm')
    plt.show()

    plt.figure()
    plt.title(f'differences in time of score by algorithm, k={k}')
    plt.bar(times_of_score.keys(), times_of_score.values())
    plt.show()

    plt.figure()
    plt.title(f'differences in scores by algorithm, k={k}')
    plt.bar(times_of_score.keys(), scores, label='score by algorithm')
    plt.show()


def zad4(X, Y):
    algorithms = ['kd_tree', 'ball_tree']
    k = 6
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

    for algorithm in algorithms:
        times_of_fit = {}
        times_of_score = {}
        scores = []

        for leaf_size in range(1, 60):
            neigh = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, leaf_size=leaf_size)

            time1 = time.time()
            neigh.fit(X, Y)
            time2 = time.time()
            times_of_fit[leaf_size] = time2 - time1

            time1 = time.time()
            score = neigh.score(X_test, y_test)
            time2 = time.time()
            times_of_score[leaf_size] = time2 - time1
            scores.append(score)

        plt.figure()
        plt.title(f'differences by leaf size for algorithm={algorithm}, k={k}')
        plt.xlabel('leaf size')
        plt.ylabel('time')
        plt.plot(times_of_fit.keys(), times_of_fit.values(), label='time of fit by leaf size')
        plt.plot(times_of_score.keys(), times_of_score.values(), label='time of score by leaf size')
        plt.show()

        plt.figure()
        plt.title(f'differences in scores by leaf size for algorithm={algorithm}, k={k}')
        plt.xlabel('leaf size')
        plt.ylabel('score')
        plt.scatter(times_of_score.keys(), scores, label='score by leaf size')
        plt.show()


def zad5(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    # k = [1, 3, 11, 25]
    k = [3]

    for val_k in k:
        neigh = KNeighborsClassifier(n_neighbors=val_k)
        neigh.fit(X_train, y_train)

        plt.figure()
        plt.scatter(neigh.predict(X_test), y_test)
        plt.show()


if __name__ == '__main__':
    [X, Y] = generate_chessboard(25)

    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()

    zad2(X, Y)

