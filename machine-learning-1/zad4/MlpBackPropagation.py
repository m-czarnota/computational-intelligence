import numpy as np

from LinearClassifier import LinearClassifier


class MlpBackPropagation(LinearClassifier):
    def __init__(self, layers: int = 2, neurons_hidden_count: int = 100, max_iter: int = 1000, alpha: float = 0.01):
        super().__init__()

        self.neurons_hidden_count = neurons_hidden_count
        self.max_iter = max_iter
        self.alpha = alpha

        self.weights_scale = 10 ** -3

        self.hidden_weights = None
        self.hidden_weights_derivatives = None

        self.output_weights = None
        self.output_weights_derivatives = None

        self.hidden_biases = None
        self.hidden_biases_derivatives = None

        self.output_biases = None
        self.output_biases_derivatives = None

    def fit(self, x: np.array, d: np.array):
        self.hidden_weights = np.random.randn(self.neurons_hidden_count, x.shape[0]) * self.weights_scale
        self.hidden_biases = np.zeros(4)

        self.output_weights = np.random.randn(d.shape[0], self.neurons_hidden_count) * self.weights_scale
        self.output_biases = np.zeros(d.shape[0])

        for _ in range(self.max_iter):
            activations, output_activations = self.forward_propagation(x)
            gradients = self.back_propagation(x, d, activations, output_activations)

    def forward_propagation(self, x: np.array):
        pre_activations = np.dot(self.hidden_weights, x) + self.hidden_biases
        activations = self.sigmoid(pre_activations)

        output_pre_activations = np.dot(self.output_weights, activations) + self.output_biases
        output_activations = self.sigmoid(output_pre_activations)

        return activations, output_activations

    def back_propagation(self, x: np.array, d: np.array, activations: np.array, output_activations: np.array):
        attr_count = x.shape[1]
        inverse_attr_count = 1 / attr_count

        output_activations_derivatives = output_activations - d
        activations_derivatives = self.output_weights_derivatives.T.dot(output_activations_derivatives) * (1 - activations ** 2)

        self.output_weights_derivatives = inverse_attr_count * output_activations_derivatives.dot(activations.T)
        self.output_biases_derivatives = inverse_attr_count * np.sum(output_activations_derivatives, axis=1, keepdims=True)

        self.hidden_weights_derivatives = inverse_attr_count * activations_derivatives.dot(x.T)
        self.hidden_biases_derivatives = inverse_attr_count * np.sum(activations_derivatives, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
