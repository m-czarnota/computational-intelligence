# poczytać o reprezentacji macierzy rzadkich w komputerach - nie ma jednego sposobu prezentacji
# policzyć częstość atrybutu, tabelka z funkcjami które policzą rozkłady brzegowe. czyli prawdopodobieńtwo łączne

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
    counts = {'x': {}, 'y': {}}
    uniques = {'x': [], 'y': []}

    for index, val_x in enumerate(x):
        val_y = y[index]

        if val_x not in uniques['x']:
            uniques['x'].append(val_x)
        if val_y not in uniques['y']:
            uniques['y'].append(val_y)

        if val_x in counts['x'].keys():
            counts['x'][val_x] += 1
        else:
            counts['x'][val_x] = 1

        if val_y in counts['y'].keys():
            counts['y'][val_y] += 1
        else:
            counts['y'][val_y] = 1

    second_returns = counts.copy()
    if prob is True:
        pass

    return [x.unique(), y.unique(), ]


def entropy(p):
    pass


if __name__ == '__main__':
    autos = pd.read_csv('autos.csv')
    print(freq(autos['horsepower']))
