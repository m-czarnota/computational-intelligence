import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from LinearClassifier import LinearClassifier
from Svm2 import Svm2


class SvmTest:
    def __init__(self, c: float = 1e-3):
        self.c_ = c

    def experiment(self):
        sonar_data = pd.read_csv('sonar_csv.csv')
        sonar_y = self.normalize_decisions(sonar_data[sonar_data.columns[-1]])
        sonar_x = sonar_data.drop(sonar_data.columns[-1], axis=1).to_numpy()

        datasets = {
            # 'linear_separable': self.generate_linear_separable_dataset(),
            'linear_non_separable': self.generate_linear_non_separable_dataset(),
            # 'sonar': [sonar_x, sonar_y],
        }

        for title, dataset in datasets.items():
            print(f'\n------ Dataset {title} ------')
            x, y = dataset

            for method in [self.svm_my_experiment, self.svm_sklearn_experiment, self.linear_regression_sklearn_experiment]:
                method(x, y, title)

    def svm_my_experiment(self, x: np.array, y: np.array, dataset_title: str):
        svm2 = Svm2(c=self.c_)

        self.fit_measure(svm2, x, y)
        self.predict_measure_and_visualise(svm2, x, y, dataset_title)

        print(f'Separation margin for {svm2}: {svm2.margin(x, y)}')
        print(f'Support vectors for {svm2}\n{svm2.sv_indexes_}')

    def svm_sklearn_experiment(self, x: np.array, y: np.array, dataset_title: str):
        svc = SVC(C=self.c_)
        self.fit_measure(svc, x, y)
        self.predict_measure_and_visualise(svc, x, y, dataset_title)

        print(f'Support vectors for {svc}\n{svc.support_vectors_}')

    def linear_regression_sklearn_experiment(self, x: np.array, y: np.array, dataset_title: str):
        linear_regression = LinearRegression()
        self.fit_measure(linear_regression, x, y)
        self.predict_measure_and_visualise(linear_regression, x, y, dataset_title)

    @staticmethod
    def fit_measure(clf, x, y):
        t1 = time.time()
        clf.fit(x, y)
        t2 = time.time()
        print(f'Time of fitting for {clf}: {t2 - t1}s')

    @staticmethod
    def predict_measure_and_visualise(clf, x, y, dataset_title: str):
        t1 = time.time()
        LinearClassifier.plot_class_universal(clf, x, y, dataset_title=dataset_title)
        t2 = time.time()
        print(f'Time of plotting with predicting for {clf}: {t2 - t1}s')

    @staticmethod
    def generate_simple_separable_dataset():
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, -1, -1, 1])

        return [x, y]

    @staticmethod
    def generate_linear_separable_dataset(n: int = 1000):
        w, b = [1, 1], -1
        x = np.random.randn(100, 2)
        d = np.sign(x.dot(w) + b)

        return [x, d]

    @staticmethod
    def generate_linear_non_separable_dataset(n: int = 1000):
        x = np.vstack((np.random.randn(n, 2), np.random.randn(n, 2) + 2))
        y = np.concatenate((np.ones(n), -np.ones(n)))

        return [x, y]

    @staticmethod
    def normalize_decisions(d):
        d_normalized = np.ones(d.shape[0]).astype("int8")
        d_normalized[d == np.unique(d)[0]] = -1

        return d_normalized
