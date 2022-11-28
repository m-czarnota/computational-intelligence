import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hmmlearn import hmm

IMAGES_DIR = './images'
LOGS_DIR = './logs'


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


def markov_experiment(x: np.array):
    n_components = [2, 3, 4, 5]

    for n in n_components:
        model = hmm.GaussianHMM(n_components=n, covariance_type='spherical')
        model.fit(x)
        X, Z = model.sample(len(x))

        predicted = model.predict(X)
        bins = sorted(df['Price_change'].unique())

        with open(f'{LOGS_DIR}/markov_experiment_states_{n}.txt', 'w') as file:
            file.write(f'-------------- state number: {n} --------------\n\n')
            file.write(f'uniques:\n{bins}\n\n')
            file.write(f'starting probabilities:\n{model.startprob_}\n\n')
            file.write(f'transition matrix:\n{model.transmat_}\n\n')
            file.write(f'means for gauss distribution:\n{model.means_}\n\n')
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
    markov_experiment(column)

    """
    początkowe prawdopodobieństwa:   
        najbardziej prawdopodobne stany według liczby stanów:
            2 - drugi = 1.00000000e+00
            3 - drugi = 1.00000000e+000
            4 - czwarty = 9.69545568e-01
            5 - trzeci = 9.99997713e-01
    
    macierze przejścia dla stanów ukrytych:
        najbardziej prawdopodobne przejścia pomiędzy stanami są dla liczby stanów 3, ze względu na zróżnicowanie, ale też nie zbyt duże zmiany, tak jak w przypadku liczby stanów 4 i 5
        
        najbardziej prawdopodobne przejścia według liczby stanów:
            2 - drugi wiersz = [0.02704475 0.97295525]
            3 - trzeci wiersz = [8.94597516e-01 2.83040466e-02 7.70984370e-02]
            4 - trzeci wiersz = [8.56806406e-01 1.11208471e-02 1.20938293e-01 1.11344535e-02]
            5 - czwarty wiersz = [7.75619325e-01 2.93375019e-03 1.10357412e-01 5.95138779e-02 5.15756352e-02]
    
    wnioski:
        najlepszą liczbą stanów wydaje się być liczba 3, ze względu na łatwą interpretację
    """
