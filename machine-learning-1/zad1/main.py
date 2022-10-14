# poczytać o reprezentacji macierzy rzadkich w komputerach - nie ma jednego sposobu prezentacji
# policzyć częstość atrybutu, tabelka z funkcjami które policzą rozkłady brzegowe. czyli prawdopodobieńtwo łączne
import math

import pandas as pd
import numpy as np


def freq(x, prob: bool = True) -> list:
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


def freq2(x, y, prob: bool = True) -> list:
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


def entropy(x, y=None):
    if y is None:
        uniques, probs = freq(x)
    else:
        uniques_x, uniques_y, probs = freq2(x, y)
    return -sum(prob * math.log2(prob) for prob in probs.values())


def infogain(x, y):
    uniques_x, uniques_y, probs = freq2(x, y)
    return entropy(x) + entropy(y) - entropy(x * y)


def kappa(x, y):
    return infogain(x, y) / entropy(y)


def gini(x, y=None, condition: bool = False):
    if y is None:
        uniques, probs = freq(x)
    else:
        uniques_x, uniques_y, probs = freq2(x, y)

    if condition is True and y is not None:
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

