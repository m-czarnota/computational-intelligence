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
    def zad2_d1():
        return df.replace('?', np.nan).dropna()

    def zad2_d2():
        df_d2 = df.copy()

        for column in df_d2.columns:
            if type(df_d2[column][0]) == str:
                uniques, counts = np.unique(df_d2[column], return_counts=True)
                value = counts[0] if counts[0] != '?' else counts[1]
                df_d2[column][df_d2[column] == '?'] = value
            else:
                value = np.mean(df_d2[column])
                df_d2[column][df_d2[column].isnull()] = value

        return df_d2

    def zad2_d3():
        df_d3 = df.copy()

        for row_i in range(df_d3.shape[0]):
            row = df_d3.iloc[row_i]
            counts = {}

            for row_value in row:
                if row_value not in counts.keys():
                    counts[row_value] = 0
                    continue

                counts[row_value] += 1

            counts = list(counts.keys())
            value_numeric = np.mean(row[row.apply(lambda x: type(x) != str)])
            value_nominal = counts[0] if counts[0] != '?' else counts[1]

            for value_i, value in enumerate(row):
                if value == '?':
                    df_d3.at[row_i, df_d3.columns[value_i]] = value_nominal
                if type(value) != str and np.isnan(value):
                    df_d3.at[row_i, df_d3.columns[value_i]] = value_numeric

        return df_d3

    data, meta = arff.loadarff(f'{DATA_FOLDER}/labor.arff')
    df = pd.DataFrame(data)
    df = df.applymap(lambda x: str(x, encoding='utf-8') if type(x) == bytes else x)
    meta = pd.DataFrame(meta)

    count_data = data.shape[0]
    count_of_empty = df.apply(lambda x: x == '?').any(1).sum()
    print('count_data:', count_data, '| count_of_empty:', count_of_empty)

    count_of_missed_values_by_column = pd.Series({column: df[column].apply(lambda x: x == '?').where(lambda x: x > 0).count() for column in df.columns})
    print('missed values by column:', count_of_missed_values_by_column[count_of_missed_values_by_column > 0])

    print(zad2_d2())


if __name__ == '__main__':
    zad2()


