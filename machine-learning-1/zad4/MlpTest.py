import numpy as np
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time

from LinearClassifier import LinearClassifier
from MlpBackPropagation import MlpBackPropagation
from MlpExtreme import MlpExtreme


class MlpTest:
    def __init__(self):
        self.layers = [1]
        self.neurons_count_in_one_hidden_layer = [100, 300, 1000]
        self.train_data_size = [0.5, 0.6, 0.8]
        self.alpha = [10 ** i for i in range(-5, 3)]

    def experiment(self):
        datasets_n = [1000, 10000, 100000]

        for n in datasets_n:
            experiment_datasets = {
                'circles': self.generate_circles_dataset(n),
                'spirals': self.generate_spirals_dataset(n),
                'chessboard': self.generate_chessboard_dataset(n),
            }

            for title, (x, y) in experiment_datasets.items():
                self.dataset_experiment((f'{title}_n_{n}', x, y))

    def dataset_experiment(self, dataset: tuple):
        for layers in self.layers:
            for neurons_count_in_one_hidden_layer in self.neurons_count_in_one_hidden_layer:
                for train_data_size in self.train_data_size:
                    neurons = tuple([neurons_count_in_one_hidden_layer for _ in range(layers)])

                    for alpha in self.alpha:
                        self.mlp_experiment(neurons, train_data_size, alpha, dataset)

    def mlp_experiment(self, neurons_in_hidden_layers: tuple, train_data_size: float, alpha: float, dataset: tuple):
        dataset_title, x, y = dataset
        dataset_title += f'_train_size_{train_data_size}'
        divided_train_test = train_test_split(x, y, train_size=train_data_size, random_state=0)

        for method in [self.mlp_sklearn_experiment, self.mlp_backprop_experiment, self.mlp_extreme_experiment]:
            print('')
            method(neurons_in_hidden_layers, divided_train_test, alpha, dataset_title)

    def mlp_sklearn_experiment(self, neurons_in_hidden_layers, divided_train_test: tuple, alpha: float, dataset_title: str):
        mlp = MLPClassifier(neurons_in_hidden_layers, alpha=alpha, max_iter=1000)
        self.experiment_for_specific_mlp(mlp, divided_train_test, dataset_title)

    def mlp_backprop_experiment(self, neurons_in_hidden_layers: tuple, divided_train_test: tuple, alpha: float, dataset_title: str):
        mlp = MlpBackPropagation(neurons_in_hidden_layers[0], max_iter=1000, alpha=alpha)
        self.experiment_for_specific_mlp(mlp, divided_train_test, dataset_title)

    def mlp_extreme_experiment(self, neurons_in_hidden_layers: tuple, divided_train_test: tuple, alpha: float, dataset_title: str):
        mlp = MlpExtreme(LogisticRegression(max_iter=1000), neurons_in_hidden_layers[0])
        self.experiment_for_specific_mlp(mlp, divided_train_test, dataset_title)

    def experiment_for_specific_mlp(self, mlp, divided_train_test: tuple, dataset_title: str):
        x_train, x_test, y_train, y_test = divided_train_test

        t1 = time.time()
        mlp.fit(x_train, y_train)
        t2 = time.time()
        print(f'Time of fitting on {dataset_title} for {mlp}: {t2 - t1}s')

        self.classification_quality(mlp, x_train, y_train, 'train')
        self.classification_quality(mlp, x_test, y_test, 'test')

        LinearClassifier.plot_class_universal(mlp, x_test, y_test, dataset_title=dataset_title)

    @staticmethod
    def classification_quality(mlp, x: np.array, y: np.array, classification_type: str = 'test'):
        t1 = time.time()
        y_pred = mlp.predict(x)
        t2 = time.time()
        print(f'Time of {mlp} prediction for {classification_type} data: {t2 - t1}s')

        print(f'Metrics of {mlp} for {classification_type}:')
        accuracy = metrics.accuracy_score(y, y_pred)
        print(f'\tAccuracy: {accuracy}')

        f1_score = metrics.f1_score(y, y_pred)
        print(f'\tF1: {f1_score}')

        auc_score = metrics.roc_auc_score(y, y_pred)
        print(f'\tAUC: {auc_score}')

    @staticmethod
    def generate_chessboard_dataset(n: int = 1000, m: int = 3):
        x = np.random.rand(n, 2) * m
        y = np.mod(np.sum(np.floor(x), axis=1), 2) * 2. - 1.
        x = x + np.random.randn(*x.shape) * 0.1

        return x, y

    @staticmethod
    def generate_circles_dataset(n: int = 1000):
        x, y = datasets.make_circles(n)

        return x, y

    @staticmethod
    def generate_spirals_dataset(n: int = 100, noise: float = 0.1, length: int = 2):
        t = np.linspace(0, (2 * np.pi * length) ** 2, n // 2)
        t = t ** 0.5

        x1 = (0.2 + t) * np.cos(t)
        y1 = (0.2 + t) * np.sin(t)
        x2 = (0.2 + t) * np.cos(t + np.pi)
        y2 = (0.2 + t) * np.sin(t + np.pi)

        x = np.array([np.concatenate((x1, x2)), np.concatenate((y1, y2))]).T + np.random.randn(n, 2) * noise
        y = np.concatenate((-np.ones(n // 2), +np.ones(n // 2)))
        p = np.random.permutation(n)

        x = x[p, :]
        y = y[p]

        return x, y
