import time
from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor, HuberRegressor, Lasso
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class RegressionClassifierTest:
    def __init__(self, noise_factor: float = 0.2, max_iter: int = 1000):
        self.noise_factor: float = noise_factor
        self.max_iter: int = max_iter

    def experiment(self, dataset_type: str = 'linear', strong_noise: bool = False, dataset_size: int = 100, polynomial_degree: int = 3):
        dataset_params = {'size': dataset_size, 'strong_noise': strong_noise}
        if dataset_type != 'linear':
            dataset_params['polynomial_degree'] = polynomial_degree

        x, y = self.generate_linear_dataset(**dataset_params) if dataset_type == 'linear' else self.generate_non_linear_dataset(**dataset_params)

        x_train, x_test, y_train, y_test = train_test_split(x, y)
        regressors = self.get_regressors_for_linear_experiment() if dataset_type == 'linear' else self.get_regressors_for_non_linear_experiment()

        plt.figure()
        plt.scatter(x, y)

        for regressor in regressors:
            self.experiment_for_regressor(regressor, data=(x, y), plot=True)

        plt.legend()
        plt.show()

    @staticmethod
    def experiment_for_regressor(regressor: RegressorMixin, data: tuple, plot: bool = True):
        model = make_pipeline(PolynomialFeatures(15), regressor)
        scaler = StandardScaler()

        x, y = list(map(lambda coordinates: scaler.fit_transform(coordinates.reshape(-1, 1)), data))

        fit_time1 = time.time()
        model.fit(x, y)
        fit_time2 = time.time()
        fit_time = fit_time2 - fit_time1

        if plot:
            y_predicted = model.predict(x)
            mse = mean_squared_error(y_predicted, y)
            plt.plot(x, y_predicted, label=regressor.__class__.__name__)

    def generate_linear_dataset(self, size: int = 100, strong_noise: bool = False) -> Tuple:
        x = np.linspace(-2, 2, size)
        y = np.vectorize(self.generate_noise)(np.linspace(-2, 2, size))

        if strong_noise:
            y = self.add_strong_noise(y)

        return x, y

    def generate_non_linear_dataset(self, size: int = 100, strong_noise: bool = False, polynomial_degree: int = 3) -> Tuple:
        x = np.linspace(-2, 2, size)

        y_fragments = [x ** degree for degree in range(polynomial_degree, -1, -1)]
        y = y_fragments[0]

        for y_fragment in y_fragments[1:]:
            y += y_fragment

        y = np.vectorize(self.generate_noise)(y)

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

    def get_regressors_for_linear_experiment(self) -> list:
        return [
            LinearRegression(),
            HuberRegressor(epsilon=5, max_iter=self.max_iter),
            Ridge(max_iter=self.max_iter),
            ElasticNet(max_iter=self.max_iter),
            # SGDRegressor(penalty='l1', max_iter=self.max_iter),
            # Lasso(max_iter=self.max_iter),
        ]

    def get_regressors_for_non_linear_experiment(self) -> list:
        return [
            LinearRegression(),
            HuberRegressor(epsilon=5, max_iter=self.max_iter),
            Ridge(),
            ElasticNet(),
            # SGDRegressor(penalty='l2', max_iter=self.max_iter),
            # Lasso(max_iter=self.max_iter),
            # MLPRegressor(),
        ]
