# poczytać o reprezentacji macierzy rzadkich w komputerach - nie ma jednego sposobu prezentacji
# policzyć częstość atrybutu, tabelka z funkcjami które policzą rozkłady brzegowe. czyli prawdopodobieńtwo łączne
import math

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_rcv1
import time
from scipy import sparse


def freq(x, prob: bool = True) -> list:
    if type(x) == sparse.csr_matrix:
        x = x.data

    counts = {}
    uniques = []

    for val in x:
        if val not in uniques:
            uniques.append(val)

        if val in counts.keys():
            counts[val] += 1
            continue

        counts[val] = 1

    total = sum(counts.values())
    return [uniques, counts if prob is False else {key: val / total for key, val in counts.items()}]


def freq2(x, y, prob: bool = True):
    if type(x) == sparse.csr_matrix or type(x) == sparse.csc_matrix:
        return freq2_sparse(x, y, prob)

    counts = {}
    uniques = {'x': [], 'y': []}

    for x_val in x:
        if x_val not in uniques['x']:
            uniques['x'].append(x_val)

        for y_val in y:
            key = (x_val, y_val)

            if key not in counts.keys():
                counts[key] = 1

                if y_val not in uniques['y']:
                    uniques['y'].append(y_val)
            else:
                counts[key] += 1

    total = sum(counts.values())
    return [uniques['x'], uniques['y'], counts if prob is False else {key: val / total for key, val in counts.items()}]


def freq2_sparse(x, y, prob: bool = True):
    """
    zrobić set(X), set(Y). cześć wspólna jako set.intersection(set)
    """
    x_nonzero = x.nonzero()[0]
    y_nonzero = y.nonzero()[0]

    uniques_x = set(x_nonzero)
    uniques_y = set(y_nonzero)
    intersection_x_y = uniques_x.intersection(uniques_y)

    count_intersection = len(intersection_x_y)
    count_x_nonzero = len(x_nonzero)
    count_y_nonzero = len(y_nonzero)
    count_shared_zeros = x.shape[0] - count_x_nonzero + y.shape[0] - count_y_nonzero

    distribution_table = [
        [count_shared_zeros, count_y_nonzero - count_intersection],
        [count_x_nonzero - count_intersection, count_intersection]
    ]
    print(distribution_table)

    return uniques_x, uniques_y, distribution_table if prob is False else [np.array(var) / (x.shape[0] + y.shape[0]) for var in distribution_table]


def entropy(x, y=None, conditional_reverse: bool = False):
    if y is None:
        uniques, probs = freq(x)
    else:
        uniques_x, uniques_y, probs = freq2(x, y)

        if conditional_reverse is True and y is not None:
            uniques_x, probs_x = freq(x)
            entropy_y = entropy(y)

            return sum(prob * entropy_y for prob in probs_x.values())

    return -sum(prob * math.log2(prob) for prob in probs.values())


def infogain(x, y, reverse: bool = False):
    if reverse is False:
        return entropy(x) + entropy(y) - entropy(x * y)
    return entropy(y) - entropy(x, y, conditional_reverse=True)


def kappa(x, y):
    return infogain(x, y) / entropy(y)


def gini(x, y=None, conditional_reverse: bool = False):
    if y is None:
        uniques, probs = freq(x)
    else:
        uniques_x, uniques_y, probs = freq2(x, y)

    if conditional_reverse is True and y is not None:
        uniques, probs = freq(x)
        gini_y = gini(y)
        return sum(prob * gini_y for prob in probs.values())

    return 1 - sum(prob ** 2 for prob in probs.values())


def ginigain(x, y):
    return gini(y) - gini(x, y, True)


if __name__ == '__main__':
    autos = pd.read_csv('zoo.csv')
    info_gains = {key: entropy(autos[key]) for key in autos.columns}
    print(sorted(info_gains.items(), key=lambda x: x[1], reverse=True))

    rcv1 = fetch_rcv1()
    X = rcv1['data'] > 0
    Xr = X[:, 2]
    # print(X[:, 1], X[1, :])
    Y = rcv1['target'][:, 87]

    uniques_x, uniques_y, probs = freq2(Xr, Y)
    print(entropy(Xr, Y))

    # print(type(Y), Y.data)
    # print(type(Y[260, 0]), Y[260, 0])

    # print(rcv1.keys())
    # print(rcv1['DESCR'])
    # print(rcv1['target_names'])
    # print(rcv1['sample_id'].shape)

    """
    wybrać jednego targeta, wybrać pojęcie które się zna. to będzie zmienna Y
    zmienne warunkowe są w pliku data w zmiennej X.
    słowa wczytać: word_list = pd.read_csv(..., delimiter=" ")
    """

    word_list = pd.read_csv('stem.termid.idf.map.txt', sep=' ')
    # print(word_list['A'])

