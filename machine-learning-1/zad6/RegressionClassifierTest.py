from typing import Tuple

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Huber, Ridge, ElasticNet, SGDRegressor, HuberRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class RegressionClassifierTest:
    def __init__(self):
        self.noise_factor: float = 0.2

    def experiment(self):
        self.experiment_for_liner_dataset()
        self.experiment_for_non_linear_dataset()

    def experiment_for_liner_dataset(self):
        x, y = self.generate_linear_dataset()
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        regressors = self.get_regressors_for_linear_experiment()

        plt.figure()
        plt.scatter(x_train, y_train)

        for regressor in regressors:
            regressor.fit(x_train, y_train)
            y_predicted = regressor.predict(x_test)
            plt.plot(x_test, y_predicted, label=regressor.__class__.__name__)

        plt.legend()
        plt.show()

    def experiment_for_non_linear_dataset(self):
        x, y = self.generate_non_linear_dataset()

    def experiment_for_regressor(self, regressor: RegressorMixin):
        ...

    def generate_linear_dataset(self, size: int = 100, strong_noise: bool = False) -> Tuple:
        """
        0 średnia
        odchylenie standardowe jakieś przyjąć

        1-wymiarowy x i 1 wymiarowy y
        """
        x_cords = np.vectorize(self.generate_noise)(np.linspace(-2, 2, size))
        y_cords = np.vectorize(self.generate_noise)(np.linspace(-2, 2, size))

        x = np.array([x_cords, y_cords]).reshape((x_cords.size, 2))

        if strong_noise:
            sample_indexes = np.random.randint(x.shape[0], size=np.random.randint(5, 11))
            for sample_index in sample_indexes:
                x[1, sample_index] += np.random.randint(-20, 20)

        return x, np.sign(y_cords) > 0

    def generate_non_linear_dataset(self) -> Tuple:
        """
        20-wymiarowy x (x^0, x^1, x^2, ...)
        """
        ...

    def generate_noise(self, data: np.array) -> np.array:
        return data + ((np.random.random() * 2) - 1) * self.noise_factor

    @staticmethod
    def get_regressors_for_linear_experiment() -> list:
        return [
            LinearRegression(),
            HuberRegressor(),
            Ridge(),
            ElasticNet(),
            SGDRegressor(),
        ]
