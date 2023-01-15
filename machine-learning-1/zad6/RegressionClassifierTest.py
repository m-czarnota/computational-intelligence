from typing import Tuple
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Huber, Ridge, ElasticNet, SGDRegressor, HuberRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class RegressionClassifierTest:
    def __init__(self):
        self.noise_factor: float = 0.2

    def experiment(self, dataset_type: str = 'linear', strong_noise: bool = False, dataset_size: int = 100, polynomial_degree: int = 3):
        dataset_params = {'size': dataset_size, 'strong_noise': strong_noise}
        if dataset_type != 'linear':
            dataset_params['polynomial_degree'] = polynomial_degree

        x, y = self.generate_linear_dataset(**dataset_params) if dataset_type == 'linear' else self.generate_non_linear_dataset(**dataset_params)

        x_train, x_test, y_train, y_test = train_test_split(x, y)
        regressors = self.get_regressors_for_linear_experiment() if dataset_type == 'linear' else self.get_regressors_for_non_linear_experiment()

        plt.figure()
        plt.scatter(x_train, y_train)

        for regressor in regressors:
            self.experiment_for_regressor(regressor, data=(x_train, x_test, y_train), plot=True)

        plt.legend()
        plt.show()

    @staticmethod
    def experiment_for_regressor(regressor: RegressorMixin, data: tuple, plot: bool = True):
        x_train, x_test, y_train = data
        regressor.fit(x_train.reshape(-1, 1), y_train)

        if plot:
            y_predicted = regressor.predict(x_test.reshape(-1, 1))
            plt.plot(x_test, y_predicted, label=regressor.__class__.__name__)

    def generate_linear_dataset(self, size: int = 100, strong_noise: bool = False) -> Tuple:
        x = np.linspace(-2, 2, size)
        y = np.vectorize(self.generate_noise)(np.linspace(-2, 2, size))

        if strong_noise:
            y = self.add_strong_noise(y)

        return x, y

    def generate_non_linear_dataset(self, size: int = 100, strong_noise: bool = False, polynomial_degree: int = 3) -> Tuple:
        x = np.linspace(-2, 2, size)
        y = np.vectorize(self.generate_noise)(x ** polynomial_degree)

        if strong_noise:
            y = self.add_strong_noise(y)

        return x, y

    def generate_noise(self, data: np.array) -> np.array:
        return data + ((np.random.random() * 2) - 1) * self.noise_factor

    @staticmethod
    def add_strong_noise(data: np.array) -> np.array:
        sample_indexes = np.random.randint(data.shape[0], size=np.random.randint(5, 11))
        for sample_index in sample_indexes:
            data[sample_index] += np.random.randint(-20, 20)

        return data

    @staticmethod
    def get_regressors_for_linear_experiment() -> list:
        return [
            LinearRegression(),
            HuberRegressor(),
            Ridge(),
            ElasticNet(),
            SGDRegressor(),
        ]

    @staticmethod
    def get_regressors_for_non_linear_experiment() -> list:
        return [
            LinearRegression(),
            HuberRegressor(),
            Ridge(),
            ElasticNet(),
            SGDRegressor(),
        ]
