import time
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor, HuberRegressor, Lasso
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from StopWatch import StopWatch


class RegressionClassifierTest:
    def __init__(self, noise_factor: float = 0.2, max_iter: int = 1000):
        self.noise_factor: float = noise_factor
        self.max_iter: int = max_iter

        self.results = None
        self.__stop_watch = StopWatch()

    def experiment(self, dataset_type: str = 'linear', strong_noise: bool = False, dataset_size: int = 100, polynomial_degree: int = 3) -> None:
        dataset_params = {'size': dataset_size, 'strong_noise': strong_noise}
        if dataset_type != 'linear':
            dataset_params['polynomial_degree'] = polynomial_degree

        self.results = pd.DataFrame(columns=['fit time', 'predict time', 'mse', 'mae'])

        x, y = self.generate_linear_dataset(**dataset_params) if dataset_type == 'linear' else self.generate_non_linear_dataset(**dataset_params)
        regressors = self.get_regressors_for_linear_experiment() if dataset_type == 'linear' else self.get_regressors_for_non_linear_experiment()

        plt.figure()
        plt.scatter(x, y)

        for regressor in regressors:
            results = self.experiment_for_regressor(regressor, data=(x, y), plot=True)
            self.results = pd.concat([self.results, results.to_frame().T], ignore_index=True)

        self.results.index = [regressor.__class__.__name__ for regressor in regressors]

        plt.legend()
        plt.show()

    def experiment_for_regressor(self, regressor: RegressorMixin, data: tuple, plot: bool = True) -> pd.Series:
        model = make_pipeline(PolynomialFeatures(15), regressor)
        scaler = StandardScaler()

        x, y = list(map(lambda coordinates: scaler.fit_transform(coordinates.reshape(-1, 1)), data))

        fit_time, _ = self.__stop_watch.measure(model.fit, [x, y])
        predict_time, y_predicted = self.__stop_watch.measure(model.predict, [x])

        mse = mean_squared_error(y, y_predicted)
        mae = mean_absolute_error(y, y_predicted)
        # score = model.score(x)

        if plot:
            plt.plot(x, y_predicted, label=regressor.__class__.__name__)

        return pd.Series({
            'fit time': f'{fit_time:.4f}s',
            'predict time': f'{predict_time:.4f}s',
            'mse': f'{mse:.4f}',
            'mae': f'{mae:.4f}',
        })

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
        return data + ((np.random.normal() * 2) - 1) * self.noise_factor

    @staticmethod
    def add_strong_noise(data: np.array, modifier: float = 5) -> np.array:
        min_val = np.min(data)
        max_val = np.max(data)

        sample_indexes = np.random.randint(data.shape[0], size=np.random.randint(5, 11))
        for sample_index in sample_indexes:
            data[sample_index] += np.random.randint(min_val - modifier, max_val + modifier)

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
