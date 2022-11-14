from scipy.io import arff
import pandas as pd
import numpy as np

DATA_FOLDER = './data'


def get_quality_and_relevance(x: pd.DataFrame):
    y = x[x.columns[-1]].tolist()
    x = x.drop(x.columns[-1], axis=1)

    quality_p = 0.75
    quality_current = 1

    while quality_current > quality_p:
        for column in x.columns:
            x_copy = x.drop(column, axis=1)
            counts = {}

            print(x_copy)

            for index, row_copy in x_copy.iterrows():
                print(row_copy)
                key = ''
                for value in row_copy:
                    key += str(value)

                counts[key] = y[index]

            uniques, counts = np.unique(counts, return_counts=True)
            print(uniques, counts)

            break

        break

    while quality_current > quality_p:
        for column in x.columns:
            x_copy = x.drop(column, axis=1)

            duplicates = f = df[df.duplicated(keep=False)]
            duplicates = duplicates.groupby(list(df)).apply(lambda a: tuple(a.index))
            print(duplicates)


            break

        break


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
    get_quality_and_relevance(arr)

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
