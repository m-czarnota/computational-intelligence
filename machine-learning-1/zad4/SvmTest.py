import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from LinearClassifier import LinearClassifier
from Svm2 import Svm2
from Svm2Sparse import Svm2Sparse


class SvmTest:
    def __init__(self, c: float = 1e-3):
        self.c_ = c

    def experiment(self):
        sonar_data = pd.read_csv('sonar_csv.csv')
        sonar_y = self.normalize_decisions(sonar_data[sonar_data.columns[-1]])
        sonar_x = sonar_data.drop(sonar_data.columns[-1], axis=1).to_numpy()

        datasets = {
            'linear_separable': self.generate_linear_separable_dataset(),
            'linear_non_separable': self.generate_linear_non_separable_dataset(),
            'sonar': [sonar_x, sonar_y],
        }

        for title, dataset in datasets.items():
            print(f'------ Dataset {title} ------')
            x, y = dataset

            svm2 = Svm2(c=self.c_)
            svm2_sparse = Svm2Sparse(c=self.c_)

            for clf in [svm2, svm2_sparse]:
                self.svm_my_experiment(clf, x, y)

    def svm_my_experiment(self, clf, x, y):
        self.fit_measure(clf, x, y)
        self.predict_measure_and_visualise(clf, x, y)

        print(f'Separation margin: {clf.margin(x, y)}')
        print(f'Support vectors\n{clf.sv_indexes_}')

    def svm_sklearn_experiment(self, x, y):
        svc = SVC(C=self.c_)
        self.fit_measure(svc, x, y)
        self.predict_measure_and_visualise(svc, x, y, True)

        print(f'Support vectors\n{svc.support_vectors_}')

    @staticmethod
    def fit_measure(clf, x, y):
        t1 = time.time()
        clf.fit(x, y)
        t2 = time.time()
        print(f'Time of fitting for {clf}: {t2 - t1}s')

    @staticmethod
    def predict_measure_and_visualise(clf, x, y, is_universal_plot: bool = False):
        t1 = time.time()
        LinearClassifier.plot_class_universal(clf, x, y) if is_universal_plot else clf.plot_class(x, y)
        t2 = time.time()
        print(f'Time of plotting with predicting for {clf}: {t2 - t1}s')

    @staticmethod
    def generate_linear_separable_dataset():
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, -1, -1, 1])

        return [x, y]

    @staticmethod
    def generate_linear_non_separable_dataset():
        n = 50
        x = np.vstack((np.random.randn(n, 2), np.random.randn(n, 2) + 2))
        y = np.concatenate((np.ones(n), -np.ones(n)))

        return [x, y]

    @staticmethod
    def normalize_decisions(d):
        d_normalized = np.ones(d.shape[0]).astype("int8")
        d_normalized[d == np.unique(d)[0]] = -1

        return d_normalized
