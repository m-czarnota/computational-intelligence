from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# !!circular import error of Linear Classifier!!
# from Svm1 import Svm1


class LinearClassifier(ABC):
    def __init__(self, coefs=None, intercepts=None, class_labels=None, max_seconds: int = 3600):
        self.coefs_ = coefs  # w
        self.intercepts_ = intercepts  # b
        self.class_labels_ = class_labels

        self.max_seconds = max_seconds
        self.iteration_count = 0

    @abstractmethod
    def fit(self, x, d):
        ...

    def margin(self, x, d, distance: bool = False):
        # margines w danych to jest minimum z wektora. jak margines jest dodatni, to znaczy, że jest separowalne
        margin = (x.dot(self.coefs_) + self.intercepts_) * d
        if distance:
            margin /= np.linalg.norm(self.coefs_)

        return margin

    def decision_function(self, x: np.array):
        return self.margin(x, np.ones(x.shape[0]))

    def predict_proba(self, x: np.array):
        a, b = self.margin(x, np.ones((x.shape[0], )))
        i = 1 - 1 / (1 + np.exp(-b))
        j = 1 / (1 + np.exp(-b))

        return np.array([i, j]).T

    def predict(self, x: np.array):
        results = np.sign(x.dot(self.coefs_) + self.intercepts_)
        results_mapped = self.class_labels_[1 * (results > 0)] if self.class_labels_ is not None else results

        return results_mapped

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        pass

    def __str__(self):
        return f'{self.__class__.__name__}[w={self.coefs_}, b={self.intercepts_}]'

    def plot_class(self, x, y, is_line: bool = False):
        self.plot_class_universal(self, x, y, is_line)

    @staticmethod
    def plot_class_universal(clf, x, y, is_line: bool = False):
        x1_min = np.min(x[:, 0]) - 0.5
        x1_max = np.max(x[:, 0]) + 0.5

        plt.figure()

        if is_line:
            points = np.array([[i, -(clf.coefs_[0] * i + clf.intercepts_) / clf.coefs_[1]] for i in np.linspace(x1_min, x1_max)])
            plt.plot(points[:, 0], points[:, 1], 'k')
        else:
            x2_min = np.min(x[:, 1]) - 0.5
            x2_max = np.max(x[:, 1]) + 0.5

            number_of_points = 250
            xn, yn = np.meshgrid(np.linspace(x1_min, x1_max, number_of_points), np.linspace(x2_min, x2_max, number_of_points))
            zn = clf.predict(np.c_[xn.flatten(), yn.flatten()]).reshape(xn.shape)

            plt.contourf(xn, yn, zn, cmap=ListedColormap(['y', 'r']))

        # !!circular import error of Linear Classifier!!
        # if isinstance(self, Svm1):
        #     indexes = self.sv_indexes_
        #     plt.plot(x[indexes, 0], x[indexes, 1], 'co', markersize=10, alpha=0.5)

        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=ListedColormap(['b', 'g']))
        plt.title('Boundary of separation')

        plt.show()
