import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Svm2 import Svm2


class LinearClassifierTest:
    def __init__(self):
        self.test_sizes = np.logspace(-3, -0.5)
        self.alphas = [10 ** i for i in range(-4, 0)]

        self.classifiers = [SVC(kernel='linear'), Svm2(), MLPClassifier(hidden_layer_sizes=())]
        self.classifiers.append(*[LogisticRegression(penalty=penalty) for penalty in ['l1', 'l2', 'elasticnet']])
        for penalty in ['l1', 'l2', 'elasticnet']:
            self.classifiers.append(LogisticRegression(penalty=penalty))

        self.data_table = []

    def experiment(self, x, y):

        for test_size in self.test_sizes:
            for i in range(100):
                separated_data = train_test_split(x, y, test_size=test_size, random_state=0)

                for alpha in self.alphas:
                    for clf in self.classifiers:
                        self.clf_experiment(clf, separated_data)
