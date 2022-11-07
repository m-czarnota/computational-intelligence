import pandas as pd
import numpy as np
from scipy.io import arff

DATA_FOLDER = './data'


def zad1():
    d = np.random.normal(size=(1000, 4))
    df = pd.DataFrame(d, columns=list('abcd'))
    print(df.shape)
    print(np.any(df.iloc[0] > 2))

    scope = (-2, 2)
    print(f'a) {len(df[df < scope[0]]) + len(df[df > scope[1]])}')
    print({column: len(df[df[column] < scope[0]]) + len(df[df[column] > scope[1]]) for column in df.columns})

    df_c = df.drop([index for index in range(df.shape[0]) if np.logical_or(np.any(df.iloc[index] < -2), np.any(df.iloc[index] > 2))])
    print(f'c: {df_c.shape[0]}')

    df_d = df.applymap(lambda x: x ** 2 if x < 0 else x)
    print(f'd: {df_d}')


def zad2():
    """
    każda z próbek brakujących będzie zastąpiona taką ilością próbek, ile jest atrybutów
    jak mamy ?, 1, 7, c1 - te ? mogło być nie potrzebne
    więc z wartości zbiorów {a1, a2, a3} to sobie powtarzamy dla każdej z tej wartości:
        a1, 1, 7, c1
        a2, 1, 7, c1
        a3, 1, 7, c1
    dla dwóch robimy ich wszystkie permutacje
    """
    data, meta = arff.loadarff(f'{DATA_FOLDER}/labor.arff')
    df = pd.DataFrame(data)
    df = df.applymap(lambda x: str(x, encoding='utf-8') if type(x) == bytes else x)

    count_data = data.shape[0]
    count_of_empty = df.apply(lambda x: x == '?').any(1).sum()

    print({column: df[column == df[column].apply(lambda x: x == '?')].any(1).sum() for column in df.columns})

if __name__ == '__main__':
    zad2()


