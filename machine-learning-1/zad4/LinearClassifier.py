from abc import ABC, abstractmethod
import numpy as np


class LinearClassifier(ABC):
    def __init__(self, coef=None, intercept=None, class_labels=None, max_seconds: int = 3600):
        self.coef_ = coef  # w
        self.intercept_ = intercept  # b
        self.class_labels_ = class_labels

        self.max_seconds = max_seconds
        self.iteration_count = 0

    @abstractmethod
    def fit(self, x, d):
        ...

    def margin(self, x, d, distance: bool = False):
        # margin = 1 / np.sqrt(np.sum(self.coef_ ** 2))
        # down = d - np.sqrt(1 + x ** 2) * margin
        # up = d + np.sqrt(1 + x ** 2) * margin
        #
        # return down, up

        # margines w danych to jest minimum z wektora. jak margines jest dodatni, to znaczy, że jest separowalne
        margin = (x.dot(self.coef_) + self.intercept_) * d
        if distance:
            margin /= np.linalg.norm(self.coef_)

        return margin

    def decision_function(self, x: np.array):
        return self.margin(x, np.ones(x.shape[0]))

    def predict_proba(self, x: np.array):
        a, b = self.margin(x, np.ones((x.shape[0], )))
        i = 1 - 1 / (1 + np.exp(-b))
        j = 1 / (1 + np.exp(-b))

        return np.array([i, j]).T

    def predict(self, x: np.array):
        results = np.sign(x.dot(self.coef_) + self.intercept_)
        results_mapped = self.class_labels_[1 * (results > 0)]

        return results_mapped

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        pass

    def __str__(self):
        return f'{self.__class__.__name__}[w={self.coef_}, b={self.intercept_}]'
