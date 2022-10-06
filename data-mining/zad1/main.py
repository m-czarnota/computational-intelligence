from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from statistics import mean
import numpy as np


def get_count_for_dataset(dataset, keys_to_avoid=[]):
    return {key: dataset[key].value_counts() for key in dataset.keys() if key not in keys_to_avoid}


def get_mean_for_continuous_keys_from_dataset(dataset, keys):
    return {key: pd.DataFrame.mean(dataset[key]) for key in keys}


def get_std_for_continuous_keys_from_dataset(dataset, keys):
    return {key: pd.DataFrame.std(dataset[key]) for key in keys}


def generate_hist(dataset):
    dataset.hist()
    plt.show()


def generate_box_plot(dataset):
    dataset.boxplot()
    plt.show()


def get_main_factors(array):
    copy = array.copy()
    main_factors = []

    while np.sum(main_factors) < 0.9:
        max_val = np.max(copy)
        main_factors.append(max_val)
        np.delete(max_val, np.where(copy == max_val))

    return main_factors


def do_pca_for_dataset(x, y):
    for n in range(2, 4):
        pca = PCA(n)
        pca.fit(x)
        X_pca = pca.transform(x)

        iris_explained_variance_ratio = pca.explained_variance_ratio_
        print(f'explained_variance_ratio_ n_components={n}:')
        for index, val in enumerate(iris_explained_variance_ratio):
            print(str(index) + ': ', str(val))

        iris_main_factors = get_main_factors(iris_explained_variance_ratio)
        print(f'main factors n_components={n}:', iris_main_factors)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if n > 2:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
        plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    mean_iris = [mean(X[:, number]) for number in range(0, 4)]
    std_iris = [np.std(X[:, number]) for number in range(0, 4)]

    zoo = pd.read_csv('zoo.csv')
    autos = pd.read_csv('autos.csv')

    zoo_keys_continuous = ['legs']
    mean_continuous_zoo = get_mean_for_continuous_keys_from_dataset(zoo, zoo_keys_continuous)
    std_continuous_zoo = get_std_for_continuous_keys_from_dataset(zoo, zoo_keys_continuous)
    count_discrete_zoo = get_count_for_dataset(zoo, zoo_keys_continuous)

    autos_keys_continuous = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
                             'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
                             'highway-mpg', 'price', 'symboling']
    mean_continuous_autos = get_mean_for_continuous_keys_from_dataset(autos, autos_keys_continuous)
    std_continuous_autos = get_std_for_continuous_keys_from_dataset(autos, autos_keys_continuous)
    count_discrete_autos = get_count_for_dataset(autos, autos_keys_continuous)

    zoo.hist(column=zoo_keys_continuous)
    plt.show()
    zoo.boxplot(column=zoo_keys_continuous)
    plt.show()

    autos.hist(column=autos_keys_continuous)
    plt.show()
    autos.boxplot(column=autos_keys_continuous)
    plt.show()

    legs = zoo['legs'].copy()
    zoo = zoo.drop('legs', axis='columns')
    zoo = zoo.drop('animal', axis='columns')
    zoo = zoo.drop('type', axis='columns')

    do_pca_for_dataset(X, Y)
    do_pca_for_dataset(zoo, legs)
