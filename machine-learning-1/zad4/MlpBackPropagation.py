import numpy as np

from LinearClassifier import LinearClassifier


class MlpBackPropagation(LinearClassifier):
    def __init__(self, neurons_hidden_count: int = 100, max_iter: int = 1000, alpha: float = 0.01, shuffle: bool = True):
        super().__init__()

        self.neurons_hidden_count = neurons_hidden_count
        self.max_iter = max_iter
        self.alpha = alpha
        self.shuffle = shuffle

        self.weights_scale = 10 ** -3

        self.hidden_weights = None
        self.hidden_biases = None

    def fit(self, x: np.array, d: np.array):
        self.hidden_weights = np.random.randn(self.neurons_hidden_count, x.shape[1]) * self.weights_scale
        self.hidden_biases = np.zeros((self.neurons_hidden_count, 1))

        self.coefs_ = np.random.randn(d.shape[0], self.neurons_hidden_count) * self.weights_scale
        self.intercepts_ = np.zeros((d.shape[0], 1))

        for _ in range(self.max_iter):
            if self.shuffle:
                self.shuffle_weights()

            params = self.forward_propagation(x)
            params_derivatives = self.back_propagation(x, d, params)
            self.update_weights(params_derivatives)

            # if params['a2'] == d:
            #     print('dopasowanie')
            #     break

    def forward_propagation(self, x: np.array):
        z1 = self.hidden_weights.dot(x.T) + self.hidden_biases  # neuron value at hidden layer
        a1 = self.sigmoid(z1)  # activation value at output layer
        z2 = self.coefs_.dot(a1) + self.intercepts_  # neuron value at output layer
        a2 = self.sigmoid(z2)  # activation value at output layer

        return {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    def back_propagation(self, x: np.array, d: np.array, params: dict):
        samples_count = x.shape[0]

        dz2 = params['a2'] - d
        dw2 = (1 / samples_count) * dz2.dot(params['a1'].T)
        db2 = (1 / samples_count) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.multiply(self.coefs_.T.dot(dz2), 1 - np.power(params['a1'], 2))
        dw1 = (1 / samples_count) * dz1.dot(x)
        db1 = (1 / samples_count) * np.sum(dz1, axis=1, keepdims=True)

        return {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    def update_weights(self, params_derivatives: dict):
        self.hidden_weights -= self.alpha * params_derivatives['dw1']
        self.hidden_biases -= self.alpha * params_derivatives['db1']
        self.coefs_ -= self.alpha * params_derivatives['dw2']
        self.intercepts_ -= self.alpha * params_derivatives['db2']

    def shuffle_weights(self):
        np.random.shuffle(self.hidden_weights)
        np.random.shuffle(self.coefs_)

    def predict(self, x: np.array):
        a2 = self.forward_propagation(x)['a2']
        return np.round(a2).astype(np.int)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
