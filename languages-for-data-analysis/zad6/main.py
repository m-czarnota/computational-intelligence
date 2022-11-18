from scipy.io import arff
import pandas as pd
import numpy as np

DATA_FOLDER = './data'


def calc_quality(x: pd.DataFrame, y: list):
    duplicates = x[x.duplicated(keep=False)]
    duplicates = duplicates.groupby(list(x)).apply(lambda a: tuple(a.index))

    conflicted_count = 0

    for index_row, duplicate_indexes in duplicates.items():
        decisions = [y[index] for index in duplicate_indexes]
        uniques = np.unique(decisions)

        if len(uniques) > 1:
            conflicted_count += len(uniques)

    non_conflicted_count = x.shape[0] - conflicted_count
    quality = non_conflicted_count / x.shape[0]

    return quality


def calc_quality_and_relevance(x: pd.DataFrame, y: list):
    x_quality = calc_quality(x, y)
    x_column_dropped_qualities = [calc_quality(x.drop(column, axis=1), y) for column in x.columns]
    x_relevances = [(x_quality - quality) / x_quality for quality in x_column_dropped_qualities]

    return x_column_dropped_qualities, x_relevances


def simplify_dataset(x: pd.DataFrame):
    y = x[x.columns[-1]].tolist()
    x_data = x.drop(x.columns[-1], axis=1)

    quality_p = 0.75
    quality_current = calc_quality(x_data, y)

    while quality_current > quality_p:
        print(quality_current)
        qualities, relevances = calc_quality_and_relevance(x_data, y)
        column_to_remove = x_data.columns[np.argmin(relevances)]

        x_data = x_data.drop(column_to_remove, axis=1)
        quality_current = calc_quality(x_data, y)

    return x_data, quality_current


if __name__ == '__main__':
    data, meta = arff.loadarff(f'{DATA_FOLDER}/contact-lenses.arff')
    df = pd.DataFrame(data)
    df = df.applymap(lambda x: str(x, encoding='utf-8') if type(x) == bytes else x)
    meta = pd.DataFrame(meta)

    df = df.replace('?', np.nan).dropna()
    # for column in df.columns:
    #     df[column] = pd.qcut(df[column], 4)

    arr = pd.DataFrame({
        'x1': [0, 0, 1, 1, 1, 1],
        'x2': [1, 1, 0, 1, 1, 1],
        'x3': [2, 0, 2, 0, 2, 1],
        'c': [0, 1, 1, 0, 1, 1],
    })
    arr_simplified, quality = simplify_dataset(arr)
    print(arr_simplified, quality)

"""
zad 4:
x1 x2 x3 c
0  1  2  0
0  1  0  1
1  0  2  1
1  1  0  0
1  1  2  1
1  1  1  1

jakosc = N1 / N = 6/6 = 1 (najwyższa jakość, dobry zbiór danych)
N1 - liczba niesprzecznych próbek
N - liczba wszystkich próbek

usuwamy atrybut (kolumnę) x1
jakosc = N1 / N = 2/6

usuwamy atrybut (kolumnę) x2, mamy x1 i x3
jakosc = 6/6 = 1

usuwamy atrybut (kolumnę) x3, mamy x1 i x2
jakosc = 1/6

------------------------------------------

istotność atrybutu i
ri = (jakosc - jakoscI) / jakosc

dla atrybutów:
r1 = (1 - 1/3) / 1 = 2/3
r2 = (1 - 1) / 1 = 0  -> ten można usunąć
r3 = (1 - 1/6) / 1 = 5/6

zad5:
redukt - zestaw atrybutów warunkowych, w których odrzucimy atrybuty nieistotne niewpływające na jakość klasyfikacji
redukt - okrojony zbiór atrybutów warunkowych
reguły odrzucone - bezwartościowe

powtarzamy, dopóki jakość zbioru nie spadnie poniżej jakiegoś progu gamma_p (jakosc_p)
ciagle wyliczamy jakość zbioru na nowo
im więcej będziemy usuwać atrybutów tym więcej próbek będzie sprzecznych ze sobą
uproszczamy zbiór przed wyznaczeniem tabeli decyzyjnej

zad6:
decyzja: if x1 == 1 and x2 == 0 -> c = 1 (dla usuniętego atrybutu x3)
wsparcie1 = 1, pewnosc=1

decyzja: if x1 == 1 and x2 == 1 -> c = 1
wsparcie = 1, pewnosc = 2/3

decyzja: if x1 == 0 and x2 == 1 -> c = 0
wsparcie = 1, pewnosc = 1/2 ?

decyzja: if x1 == 0 and x2 == 1 -> c = 1
wsparcie = 1, pewnosc = 1/2

decyzja: if x1 == 0 and x2 == 1 -> c = 0
wsparcie = 1, pewnosc = 1/3

najbardziej wartościowe reguły to te które mają pewność=1 i duże wsparcie
"""