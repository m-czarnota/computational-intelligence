import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hmmlearn import hmm


if __name__ == '__main__':
    df = pd.read_csv('HistoricalData_1669022749333.csv')
    df = df.applymap(lambda x: x.replace('$', '') if type(x) == str else x)
    df = df.applymap(lambda x: float(x) if str(x).isnumeric() else x)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Close/Last'] = df['Close/Last'].map(lambda x: float(x))

    df['Price_change'] = (df['Close/Last'].diff())
    df = df.drop(df[df.Date >= '11/01/2022'].index)

    # plt.figure()
    # plt.plot(df['Date'], df['Close/Last'])
    # plt.xlabel('Date')
    # plt.ylabel('Daily action price')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(df['Date'], df['Price_change'])
    # plt.xlabel('Date')
    # plt.ylabel('Daily change in price of actions')
    # plt.show()

    n_components = [1, 2, 5, 10, 25, 50, 100]
    column = df['Price_change'].to_numpy()[:, None]

    model = hmm.GaussianHMM(n_components=3, covariance_type='spherical')
    model.fit(column)
    X, Z = model.sample(len(column))

    predicted = model.predict(X)

    bins = sorted(df['Price_change'].unique())
    print(bins)

    print(model.transmat_)

    plt.figure()
    plt.plot(df['Date'], df['Close/Last'])
    plt.plot(df['Date'], predicted)
    plt.xlabel('Date')
    plt.ylabel('Daily action price')
    plt.show()

    plt.figure()
    plt.plot(df['Date'], df['Price_change'])
    plt.plot(df['Date'], predicted)
    plt.xlabel('Date')
    plt.ylabel('Daily change in price of actions')
    plt.show()




