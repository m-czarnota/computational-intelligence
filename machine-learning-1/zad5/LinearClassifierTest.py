import time
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_rcv1, fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy.sparse import csr_matrix, csc_matrix

from Svm2 import Svm2


class LinearClassifierTest:
    def __init__(self):
        self.max_iter = 1000
        self.test_sizes = np.logspace(-1, -0.5, num=2)
        self.regularization_params = [10 ** i for i in range(-3, 2)]
        self.results_count_per_test_size = 3

        penalties = [
            'l1',
            'l2',
            # 'elasticnet'  # only solver saga can handle with it, but it throws error on calc
        ]
        self.classifiers = {
            'svc': SVC(kernel='linear', max_iter=self.max_iter),
            # 'svm2': GridSearchCV(Svm2(), param_grid={'c': self.regularization_params}),
            'mlp': MLPClassifier(hidden_layer_sizes=(), max_iter=self.max_iter),
            **{f'logistic_regression_{penalty}': LogisticRegression(penalty=penalty, max_iter=self.max_iter, solver='saga') for penalty in penalties}
        }

        # uncomment to run with GridSearch. Warning!! Calculating time will drastically increase, especially for sparse data.
        # self.classifiers = {
        #     'svc': GridSearchCV(SVC(kernel='linear', max_iter=self.max_iter), param_grid={'C': self.regularization_params}),
        #     # 'svm2': GridSearchCV(Svm2(), param_grid={'c': self.regularization_params}),
        #     'mlp': GridSearchCV(MLPClassifier(hidden_layer_sizes=(), max_iter=self.max_iter), param_grid={'alpha': self.regularization_params}),
        #     **{f'logistic_regression_{penalty}': GridSearchCV(LogisticRegression(penalty=penalty, max_iter=self.max_iter, solver='saga'), param_grid={'C': self.regularization_params}) for penalty in penalties}
        # }

        self.data_table = None

    def experiment(self, verbose: bool = False):
        dataset_method_generators = {
            'sonar': self.get_sonar_data,
            # 'reuters': self.get_reuters_data,
            # 'mnist': self.get_mnist_data,
        }

        for dataset_name, method in dataset_method_generators.items():
            if verbose:
                print(f'\n--------------- {dataset_name} ---------------')

            x, y = method()

            results = self.experiment_for_dataset(x, y, verbose)
            results['dataset'] = dataset_name

            self.data_table = results if self.data_table is None else pd.concat([self.data_table, results], ignore_index=True)

    def experiment_for_dataset(self, x: np.array, y: np.array, verbose: bool = False) -> pd.DataFrame:
        mean_results_count = len(self.classifiers.keys()) * self.test_sizes.size
        mean_results_storage = [0 for _ in range(mean_results_count)]
        mean_results_iter = 0

        for test_size_iter, test_size in enumerate(self.test_sizes):
            results_storage = {clf_name: [0] * self.results_count_per_test_size for clf_name in self.classifiers.keys()}
            t1_test_size = time.time()

            for results_iter in range(self.results_count_per_test_size):
                separated_data = train_test_split(x, y, test_size=test_size)
                t1_inner_loop = time.time()

                for clf_name, clf in self.classifiers.items():
                    experiment_results = self.clf_experiment(clf, separated_data)
                    experiment_results['clf'] = clf_name

                    results_storage[clf_name][results_iter] = experiment_results

                if verbose:
                    t2_inner_loop = time.time()
                    time_inner_loop = t2_inner_loop - t1_inner_loop
                    progress_percent = (results_iter + 1) / self.results_count_per_test_size * 100

                    print(f'\t\tProgress inner loop: {progress_percent:.2f}% - {time_inner_loop:.4f}s')

            for results_iter, (clf_name, results_for_clf) in enumerate(results_storage.items()):
                results = pd.DataFrame(results_for_clf)
                mean_results = results.mean()

                mean_results['clf'] = clf_name
                mean_results['test_size'] = test_size
                mean_results_storage[mean_results_iter] = mean_results

                mean_results_iter += 1

            if verbose:
                t2_test_size = time.time()
                time_test_size = t2_test_size - t1_test_size
                progress_percent = (test_size_iter + 1) / self.test_sizes.size * 100

                print(f'\tProgress test size: {progress_percent:.2f}% - {time_test_size:.4f}s')

        return pd.DataFrame(mean_results_storage)

    def clf_experiment(self, clf, separated_data: tuple) -> pd.Series:
        x_train, x_test, y_train, y_test = separated_data

        t1_fit = time.time()
        clf.fit(x_train, y_train)
        t2_fit = time.time()
        time_fit = t2_fit - t1_fit

        classification_quality_train = self.calc_classification_quality(clf, x_train, y_train, 'train')
        classification_quality_test = self.calc_classification_quality(clf, x_test, y_test, 'test')

        return pd.Series({
            'fit_time': time_fit,
            **classification_quality_train,
            **classification_quality_test
        })

    @staticmethod
    def calc_classification_quality(clf, x: np.array, y: np.array, data_type_label: str = 'train'):
        t1_predict = time.time()
        y_predicted = clf.predict(x)
        t2_predict = time.time()
        time_predict = t2_predict - t1_predict

        accuracy = metrics.accuracy_score(y, y_predicted)
        f1_score = metrics.f1_score(y, y_predicted)
        auc_score = metrics.roc_auc_score(y, y_predicted)

        return {
            f'predict_time_{data_type_label}': time_predict,
            f'accuracy_{data_type_label}': accuracy,
            f'f1_{data_type_label}': f1_score,
            f'auc_{data_type_label}': auc_score,
        }

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

        return xr, y.toarray().ravel()

    @staticmethod
    def get_mnist_data() -> Tuple:
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")

        return csc_matrix(x)[:, 400], y  # 407 column for minimalize sparse

    @staticmethod
    def normalize_decisions(d) -> np.array:
        d_normalized = np.ones(d.size).astype("int8")
        d_normalized[d == np.unique(d)[0]] = -1

        return d_normalized
