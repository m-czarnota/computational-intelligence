from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, HuberRegressor, Lasso
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from StopWatch import StopWatch


class RegressionClassifierTest:
    def __init__(self, max_iter: int = 50000):
        self.max_iter: int = max_iter
        self.polynomial_degrees = [3, 5, 10, 15, 30, 60]

        self.results = None
        self.__stop_watch = StopWatch()

    def experiment(self, dataset_type: str = 'linear', strong_noise: bool = False, dataset_size: int = 100) -> None:
        self.results = pd.DataFrame()

        x, y = self.generate_dataset(dataset_size, dataset_type)
        y_noised = self.add_strong_noise(y, 30) if strong_noise else y

        data = {'x': x, 'y': y, 'y_noised': y_noised, 'strong_noise': strong_noise}

        for polynomial_degree in self.polynomial_degrees if dataset_type != 'linear' else [-1]:
            if dataset_type != 'linear':
                x_train, x_test = self.apply_polynomial_features(x, polynomial_degree)
                data = {**data, 'x_train': x_train, 'x_test': x_test}

            self.experiment_for_polynomial_degree(polynomial_degree, data, dataset_type)

    def experiment_for_polynomial_degree(self, deg: int, data: dict, dataset_type: str = 'linear'):
        regressors = self.get_regressors(dataset_type)
        plot_title = f'{dataset_type.capitalize()} {"with" if data["strong_noise"] else "without"} strong artificial noise'
        plot_title += f', polynomial degree = {deg}' if deg != -1 else ''

        plt.figure(figsize=(20, 10))
        plt.title(plot_title)

        plt.scatter(data['x'], data['y'], label='Train samples')
        plt.scatter(data['x'], data['y_noised'], label='Test samples', alpha=0.6)

        for regressor in regressors:
            regressor_experiment_data = (regressor, data, True)
            results = self.experiment_for_regressor(regressor_experiment_data, dataset_type)

            if dataset_type != 'linear':
                results['polynomial degree'] = deg

            self.results = pd.concat([self.results, results.to_frame().T], ignore_index=True)

        plt.legend()
        plt.show()

    def experiment_for_regressor(self, params: tuple, dataset_type: str = 'linear'):
        return self.experiment_linear_for_regressor(*params) if dataset_type == 'linear' else self.experiment_nonlinear_for_regressor(*params)

    def experiment_linear_for_regressor(self, regressor: RegressorMixin, data: dict, plot: bool = True) -> pd.Series:
        fit_time, _ = self.__stop_watch.measure(regressor.fit, [data['x'], data['y']])
        predict_time, y_predicted = self.__stop_watch.measure(regressor.predict, [data['x']])

        mse = mean_squared_error(data['y_noised'], y_predicted)
        mae = mean_absolute_error(data['y_noised'], y_predicted)

        regressor_name = regressor.__class__.__name__
        label = regressor_name + f' - {regressor.penalty}' if regressor_name == 'SGDRegressor' else regressor_name

        if plot:
            plt.plot(data['x'], y_predicted, label=label)

        return pd.Series({
            'regressor': regressor_name,
            'fit time': fit_time,
            'predict time': predict_time,
            'mse': mse,
            'mae': mae,
        })

    def experiment_nonlinear_for_regressor(self, regressor: RegressorMixin, data: dict, plot: bool = True) -> pd.Series:
        fit_time, _ = self.__stop_watch.measure(regressor.fit, [data['x_train'], data['y']])
        predict_time, y_predicted = self.__stop_watch.measure(regressor.predict, [data['x_test']])

        mse = mean_squared_error(data['y_noised'], y_predicted)
        mae = mean_absolute_error(data['y_noised'], y_predicted)

        regressor_name = regressor.__class__.__name__
        label = regressor_name + f' - {regressor.penalty}' if regressor_name == 'SGDRegressor' else regressor_name

        if plot:
            plt.plot(data['x'], y_predicted, label=label)

        return pd.Series({
            'regressor': regressor_name,
            'fit time': fit_time,
            'predict time': predict_time,
            'mse': mse,
            'mae': mae,
        })

    def generate_dataset(self, size: int, dataset_type: str = 'linear'):
        return self.generate_linear_dataset(size) if dataset_type == 'linear' else self.generate_non_linear_dataset(size)

    def generate_linear_dataset(self, size: int = 100) -> Tuple:
        x = np.sort(np.random.uniform(-2, 2, size)).reshape(-1, 1)
        y = 2 * x + 4
        y += self.generate_gauss_noise_for_data(y)

        return x, y

    def generate_non_linear_dataset(self, size: int = 100) -> Tuple:
        x = np.sort(np.random.uniform(-2, 2, size)).reshape(-1, 1)
        y = 2 * x ** 3 + 3 * x + 1
        y += self.generate_gauss_noise_for_data(y)

        return x, y

    @staticmethod
    def generate_gauss_noise_for_data(data: np.array) -> np.array:
        return np.random.normal(0, np.std(data) / 10, data.shape)

    @staticmethod
    def add_strong_noise(data: np.array, modifier: float = 5) -> np.array:
        data = np.copy(data)
        min_val = np.min(data)
        max_val = np.max(data)

        sample_indexes = np.random.randint(data.size, size=np.random.randint(0.05 * data.size, 0.11 * data.size))
        for sample_index in sample_indexes:
            data[sample_index] += np.random.randint(min_val - modifier, max_val + modifier)

        return data

    @staticmethod
    def apply_polynomial_features(data: np.array, deg: int) -> Tuple:
        scaler = StandardScaler()
        polynomial_features = PolynomialFeatures(deg)

        data_train = scaler.fit_transform(polynomial_features.fit_transform(data))
        data_test = scaler.fit_transform(polynomial_features.fit_transform(data))

        return data_train, data_test

    def get_regressors(self, dataset_type: str = 'linear'):
        return self.get_regressors_for_linear_experiment() if dataset_type == 'linear' else self.get_regressors_for_non_linear_experiment()

    def get_regressors_for_linear_experiment(self) -> list:
        return [
            LinearRegression(),
            HuberRegressor(),
            Ridge(),
            SGDRegressor(loss='huber', penalty='l1', max_iter=self.max_iter, tol=1e-6),
            SGDRegressor(loss='huber', penalty='l2', max_iter=self.max_iter, tol=1e-6),
            Lasso(alpha=0.01),
        ]

    def get_regressors_for_non_linear_experiment(self) -> list:
        return [
            LinearRegression(),
            HuberRegressor(epsilon=1, max_iter=self.max_iter),
            Ridge(),
            SGDRegressor(loss='huber', penalty='l1', max_iter=self.max_iter, tol=1e-6),
            SGDRegressor(loss='huber', penalty='l2', max_iter=self.max_iter, tol=1e-6),
            Lasso(alpha=0.01),
            MLPRegressor(),
        ]
