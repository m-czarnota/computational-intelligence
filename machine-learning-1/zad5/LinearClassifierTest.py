from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_rcv1, fetch_openml
from sklearn.model_selection import GridSearchCV

from Svm2 import Svm2


class LinearClassifierTest:
    def __init__(self):
        self.test_sizes = np.logspace(-3, -0.5)
        self.regularization_params = [10 ** i for i in range(-3, 2)]
        self.results_count_per_test_size = 100

        self.classifiers = {
            'svc': GridSearchCV(SVC(kernel='linear'), param_grid={'C': self.regularization_params}),
            'svm2': GridSearchCV(Svm2(), param_grid={'c': self.regularization_params}),
            'mlp': GridSearchCV(MLPClassifier(hidden_layer_sizes=()), param_grid={'alpha': self.regularization_params}),
            **{f'logistic_regression_{penalty}': GridSearchCV(LogisticRegression(penalty=penalty), param_grid={'C': self.regularization_params}) for penalty in ['l1', 'l2', 'elasticnet']}
        }

        self.data_table = []

    def experiment(self):
        for method in [self.get_sonar_data, self.get_reuters_data, self.get_mnist_data]:
            self.experiment_for_dataset(*method())

    def experiment_for_dataset(self, x, y):
        for test_size in self.test_sizes:
            results_storage = {clf_name: np.zeros(self.results_count_per_test_size, dtype='object') for clf_name in self.classifiers.keys()}

            for results_iter in range(results_storage.size):
                separated_data = train_test_split(x, y, test_size=test_size)

                for clf_name, clf in self.classifiers.items():
                    results_storage[clf_name][results_iter] = self.clf_experiment(clf, separated_data)

    def clf_experiment(self, clf, separated_data: tuple) -> dict:
        pass

    def get_sonar_data(self) -> Tuple:
        sonar_data = pd.read_csv('sonar_csv.csv')
        y = self.normalize_decisions(sonar_data[sonar_data.columns[-1]])
        x = sonar_data.drop(sonar_data.columns[-1], axis=1).to_numpy()

        return x, y

    @staticmethod
    def get_reuters_data() -> Tuple:
        rcv1 = fetch_rcv1()
        x = rcv1['data'] > 0
        xr = x[:, 2]
        y = rcv1['target'][:, 5]

        return xr, y

    @staticmethod
    def get_mnist_data() -> Tuple:
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")

        return x, y

    @staticmethod
    def normalize_decisions(d) -> np.array:
        d_normalized = np.ones(d.size).astype("int8")
        d_normalized[d == np.unique(d)[0]] = -1

        return d_normalized
