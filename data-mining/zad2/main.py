"""
dla każdego k powtórzyć eksperyment kilka razy i usrednić, żeby wykres nie był poszarpany
na wykresie zaznaczamy optymalną wartosć parametru k
powinno być conajmniej 1000 pkt

dzielimy zbiór na pół. y_test to wyniki dla idealnej
sprawdzać od k=1 do k=rozmiar_proby, jak rozmiar_proby == 500, to 500
dla małych k dać krok mały, dla większych dać większy - do 20 co 2, do 50 co 5 na przykład
"""


import math
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

    X = np.random.rand(CHESSBOARD_SAMPLE_COUNT, 2)

    Y = np.add(((np.floor(X[:, 0] * second_dim)) % second_dim), ((np.floor(X[:, 1] * second_dim)) % second_dim))
    Y = Y % 2

    if gauss is True:
        std = get_truncated_normal(low=0, upp=0.5)
        X[:, 0] += std.rvs()
        X[:, 1] += std.rvs()

    return X, Y


if __name__ == '__main__':
    [X, Y] = generate_chessboard(25, True)

    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()

    k_dict = {}
    k = 1
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

        if i % 10 == 0:
            k += 1

    print(max_score, k_optimal)
    for key, values in k_dict.items():
        k_dict[key] = np.mean(values)

    plt.figure()
    plt.plot(k_dict.keys(), k_dict.values())
    plt.xlabel('k')
    plt.ylabel('score')
    plt.grid()
    plt.show()

