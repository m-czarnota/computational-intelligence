from typing import TextIO

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hmmlearn import hmm

IMAGES_DIR = './images'


def get_dataset() -> pd.DataFrame:
    df = pd.read_csv('HistoricalData_1669022749333.csv')
    df = df.applymap(lambda x: x.replace('$', '') if type(x) == str else x)
    df = df.applymap(lambda x: float(x) if str(x).isnumeric() else x)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Close/Last'] = df['Close/Last'].map(lambda x: float(x))

    df['Price_change'] = (df['Close/Last'].diff())
    df = df.drop(df[df.Date >= '11/01/2022'].index)

    return df


def plot_two_charts(df: pd.DataFrame):
    plt.figure(figsize=(20, 10))
    plt.plot(df['Date'], df['Close/Last'])
    plt.xlabel('Date')
    plt.ylabel('Daily action price')
    plt.savefig(f'{IMAGES_DIR}/daily_action_price.png')
    # plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(df['Date'], df['Price_change'])
    plt.xlabel('Date')
    plt.ylabel('Daily change in price of actions')
    plt.savefig(f'{IMAGES_DIR}/daily_change_price_actions.png')
    # plt.show()


def markov_experiment(file: TextIO, x: np.array):
    n_components = [2, 3, 4, 5]

    for n in n_components:
        model = hmm.GaussianHMM(n_components=n, covariance_type='spherical')
        model.fit(x)
        X, Z = model.sample(len(x))

        predicted = model.predict(X)
        bins = sorted(df['Price_change'].unique())

        file.write(f'-------------- state number: {n} --------------\n')
        file.write(f'uniques:\n{bins}\n')
        file.write(f'starting probabilities:\n{model.startprob_}\n')
        file.write(f'transition matrix:\n{model.transmat_}\n')
        file.write(f'means for gauss distribution:\n{model.means_}\n')
        file.write(f'covariance for gauss distribution:\n{model.covars_}\n\n')

        plt.figure(figsize=(20, 10))
        plt.scatter(df['Date'], df['Close/Last'], c=predicted, s=0.2)
        plt.xlabel('Date')
        plt.ylabel('Daily action price')
        plt.savefig(f'{IMAGES_DIR}/daily_action_price_n_{n}')
        # plt.show()

        plt.figure(figsize=(20, 10))
        plt.scatter(df['Date'], df['Price_change'], c=predicted, s=0.2)
        plt.xlabel('Date')
        plt.ylabel('Daily change in price of actions')
        plt.savefig(f'{IMAGES_DIR}/daily_change_price_actions_n_{n}')
        # plt.show()


if __name__ == '__main__':
    df = get_dataset()
    plot_two_charts(df)

    column = df['Price_change'].to_numpy()[:, None]

    with open('markov_experiment.txt', 'w') as f:
        markov_experiment(f, column)
