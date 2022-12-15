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
        y = self.normalize_decisions(d, x)

        self.hidden_weights = np.random.normal(size=(x.shape[1], self.neurons_hidden_count)) * self.weights_scale
        self.hidden_biases = np.random.normal(size=self.neurons_hidden_count)

        self.coefs_ = np.random.normal(size=(self.neurons_hidden_count, y.shape[1])) * self.weights_scale
        self.intercepts_ = np.random.normal(size=y.shape[1])

        for _ in range(self.max_iter):
            if self.shuffle:
                np.random.shuffle(x)

            for x_iter, x_val in enumerate(x):
                params_activations = self.forward_propagation(x_val)
                params_fixes = self.back_propagation(x_val, y[x_iter], params_activations)
                self.update_weights(params_fixes)

    def forward_propagation(self, x: np.array):
        z1 = x.dot(self.hidden_weights) + self.hidden_biases  # neuron value at hidden layer
        a1 = self.sigmoid(z1)  # activation value at output layer
        z2 = a1.dot(self.coefs_) + self.intercepts_  # neuron value at output layer
        a2 = self.sigmoid(z2)  # activation value at output layer

        return {'a1': a1, 'a2': a2}

    def back_propagation(self, x: np.array, d: np.array, params: dict):
        delta_out = (params['a2'] - d) * (params['a2'] * (1 - params['a2']))
        gradient_out = np.outer(params['a1'], delta_out)

        delta_hidden = np.dot(delta_out, self.coefs_.T) * (params['a1'] * (1 - params['a1']))
        gradient_hidden = np.outer(x, delta_hidden)

        return {'delta_out': delta_out, 'gradient_out': gradient_out, 'delta_hidden': delta_hidden, 'gradient_hidden': gradient_hidden}

    def update_weights(self, params: dict):
        self.hidden_weights -= self.alpha * params['gradient_hidden']
        self.hidden_biases -= self.alpha * params['delta_hidden']
        self.coefs_ -= self.alpha * params['gradient_out']
        self.intercepts_ -= self.alpha * params['delta_out']

    def predict(self, x: np.array):
        a2 = self.forward_propagation(x)['a2']
        return a2

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def normalize_decisions(d: np.array, x: np.array):
        classes = np.unique(d)
        y = np.zeros((x.shape[0], classes.shape[0]))

        for i in range(x.shape[0]):
            y[i, np.where(classes == d[i])[0]] = 1

        return y
