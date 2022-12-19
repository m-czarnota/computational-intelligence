import numpy as np

from LinearClassifier import LinearClassifier


class MlpBackPropagation(LinearClassifier):
    def __init__(self, neurons_hidden_count: int = 100, max_iter: int = 1000, alpha: float = 0.01, shuffle: bool = False):
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
        self.class_labels_ = np.unique(y)

        self.hidden_weights = np.random.normal(size=(x.shape[1], self.neurons_hidden_count)) * self.weights_scale
        self.hidden_biases = np.random.normal(size=self.neurons_hidden_count)

        self.coef_ = np.random.normal(size=(self.neurons_hidden_count, y.shape[1])) * self.weights_scale
        self.intercept_ = np.random.normal(size=y.shape[1])

        for _ in range(self.max_iter):
            if self.shuffle:
                np.random.shuffle(x)

            for x_iter, x_val in enumerate(x):
                params_activations = self.forward_propagation(x_val)
                params_fixes = self.back_propagation(x_val, y[x_iter], params_activations)
                self.update_weights(params_fixes)

    def forward_propagation(self, x: np.array) -> list:
        z1 = x.dot(self.hidden_weights) + self.hidden_biases  # neuron value at hidden layer
        a1 = self.sigmoid(z1)  # activation value at output layer
        z2 = a1.dot(self.coef_) + self.intercept_  # neuron value at output layer
        a2 = self.sigmoid(z2)  # activation value at output layer

        return [a1, a2]

    def back_propagation(self, x: np.array, d: np.array, params: dict):
        a1, a2 = params

        delta_out = (a2 - d) * (a2 * (1 - a2))
        gradient_out = np.outer(a1, delta_out)

        delta_hidden = np.dot(delta_out, self.coef_.T) * (a1 * (1 - a1))
        gradient_hidden = np.outer(x, delta_hidden)

        return {'delta_out': delta_out, 'gradient_out': gradient_out, 'delta_hidden': delta_hidden, 'gradient_hidden': gradient_hidden}

    def update_weights(self, params: dict):
        self.hidden_weights -= self.alpha * params['gradient_hidden']
        self.hidden_biases -= self.alpha * params['delta_hidden']
        self.coef_ -= self.alpha * params['gradient_out']
        self.intercept_ -= self.alpha * params['delta_out']

    def predict(self, x: np.array):
        a2 = self.forward_propagation(x)['a2']
        if len(a2.shape) == 1:
            return self.class_labels_[np.argmax(a2)]

        return np.array([self.class_labels_[np.argmax(pair)] for pair in a2])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def normalize_decisions(d: np.array, x: np.array):
        classes = np.unique(d)
        y = np.full((x.shape[0], classes.shape[0]), -1)

        for i in range(x.shape[0]):
            y[i, np.where(classes == d[i])[0]] = 1

        return y

    def __str__(self):
        return f'MlpBackPropagation(neurons_hidden_count={self.neurons_hidden_count}, max_iter={self.max_iter}, alpha={self.alpha}, shuffle={self.shuffle})'
