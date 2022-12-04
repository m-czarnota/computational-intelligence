import numpy as np
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time

from LinearClassifier import LinearClassifier


class MlpTest:
    def __init__(self):
        self.layers = [1, 2, 3, 4]
        self.neurons_count_in_one_hidden_layer = [10, 30, 100, 300, 1000]
        self.train_data_size = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6]
        self.alpha = [10 ** i for i in range(-5, 3)]
        self.fit_algorithm = ['lbfgs', 'adam']

    def experiment(self):
        datasets_n = [1000, 10000, 100000]

        for n in datasets_n:
            experiment_datasets = {
                'chessboard': self.generate_chessboard_dataset(n),
                'circles': self.generate_circles_dataset(n),
                'spirals': self.generate_spirals_dataset(n),
            }

            for title, (x, y) in experiment_datasets.items():
                self.dataset_experiment((f'{title}_n_{n}', x, y))

    def dataset_experiment(self, dataset: tuple):
        for layers in self.layers:
            for neurons_count_in_one_hidden_layer in self.neurons_count_in_one_hidden_layer:
                for train_data_size in self.train_data_size:
                    for alpha in self.alpha:
                        for fit_algorithm in self.fit_algorithm:
                            self.mlp_experiment(layers, neurons_count_in_one_hidden_layer, train_data_size, alpha, fit_algorithm, dataset)

    def mlp_experiment(self, layers: int, neurons_count_in_one_hidden_layer: int, train_data_size: float, alpha: float, solver: str, dataset: tuple):
        dataset_title, x, y = dataset
        dataset_title += f'_layers_{layers}_neurons_{neurons_count_in_one_hidden_layer}_train_size_{train_data_size}_alpha_{alpha}_solver_{solver}'
        divided_train_test = train_test_split(x, y, train_size=train_data_size)

        self.mlp_sklearn_experiment(layers, neurons_count_in_one_hidden_layer, divided_train_test, alpha, solver, dataset_title)

    def mlp_sklearn_experiment(self, layers: int, neurons_count_in_one_hidden_layer: int, divided_train_test: tuple, alpha: float, solver: str, dataset_title: str):
        mlp = MLPClassifier((neurons_count_in_one_hidden_layer for i in range(layers)), solver=solver, alpha=alpha, max_iter=1000)
        self.experiment_for_specific_mlp(mlp, divided_train_test, 'MLP Classifier sklearn', dataset_title)

    def experiment_for_specific_mlp(self, mlp, divided_train_test: tuple, mlp_name: str, dataset_title: str):
        x_train, y_train, x_test, y_test = divided_train_test

        t1 = time.time()
        mlp.fit(x_train, y_train)
        t2 = time.time()
        print(f'Time of fitting on {dataset_title} for {mlp_name}: {t2 - t1}s')

        self.classification_quality(mlp, x_train, y_train, mlp_name, 'train')
        self.classification_quality(mlp, x_test, y_test, mlp_name, 'test')

        LinearClassifier.plot_class_universal(mlp, x_test, y_test)

    @staticmethod
    def classification_quality(mlp, x: np.array, y: np.array, mlp_name: str, classification_type: str = 'test'):
        t1 = time.time()
        y_pred = mlp.predict(x)
        t2 = time.time()
        print(f'Time of {mlp_name} prediction for {classification_type} data: {t2 - t1}s')

        accuracy = metrics.accuracy_score(y, y_pred)
        print(f'Accuracy of {mlp_name} for {classification_type} data: {accuracy}')

        f1_score = metrics.f1_score(y, y_pred)
        print(f'F1 of {mlp_name} for {classification_type} data: {f1_score}')

        auc_score = metrics.roc_auc_score(y, mlp.predict_proba(x))
        print(f'AUC of {mlp_name} for {classification_type} data: {auc_score}')

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
