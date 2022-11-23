from scipy.io import arff
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def forecast_course(x: pd.Series, samples_count: int, polynomial_degree: int, day: int):
    if polynomial_degree > samples_count - 1:
        raise f'k={polynomial_degree} must be lower or equal than n - 1 = {samples_count} - 1'

    # y = prognozowana_wartosc_dnia_ntego * a1 + 0.0
    t = [[t_val, 1] for t_val in np.arange(day, day - samples_count, 1)]  # 0 dzisiejszy dzień, i 'n' wcześniejszych
    y = np.array([x.iloc[d] for d in t]).T
    a = np.linalg.lstsq(t, y)


def plot_charts(df: pd.DataFrame):
    plt.figure()
    [plt.plot(df.index, df[column], label=column) for column in df.columns]
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Company stock quotes')
    plt.show()

    plt.figure()
    [df[column].rolling(25).mean().plot(label=column) for column in df.columns]
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data, meta = arff.loadarff('stock.arff')
    df = pd.DataFrame(data, index=pd.date_range('1988-01-01', periods=len(data)))
    df = df.applymap(lambda x: str(x, encoding='utf-8') if type(x) == bytes else x)
    meta = pd.DataFrame(meta)

    print(df['company1'])
    print(df['company1'].iloc[0])

    # plot_charts(df)

    # correlation = df.corr().abs().unstack().sort_values(kind='quicksort')
    # print(correlation)

    forecast_course(df['company1'], 5, 2, 6)

"""
X = np.liang.lstsq(A, B) - zwraca te mnożenie macierzy A z końca zad 6

zad6:
y = at + b
y = [a1 a0] * [t 1].T
T = [t[0] 1; t[-1] 1; t[-2] 1; ...].T
Y = [y[0]; y[-1]; y[-2]; ...].T
A = [a1, a0].T
A = (T.T * T) ** -1 * T.T * Y

prognozowanie tylko dla jednej kolumny
najlepiej żeby zadawać parametr n (ilość próbek), rząd wielomianu i zadany dzień
"""
