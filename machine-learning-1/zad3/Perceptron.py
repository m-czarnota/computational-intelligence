import numpy as np

from LinearClassifier import LinearClassifier
import time


class Perceptron(LinearClassifier):
    def __init__(self, gamma: float = 0.0, max_seconds: int = 5, **kwargs):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.max_seconds = max_seconds

        self.iteration_count = 0

    def fit(self, x: np.array, d: np.array):
        self.class_labels_ = np.unique(d)

        d_normalized = np.ones(x.shape[0]).astype("int8")
        d_normalized[d == self.class_labels_[0]] = -1

        w, b = np.ones(x.shape[1]), 0
        n = 0
        t1 = time.time()

        while n < x.shape[0]:
            t2 = time.time() - t1
            if t2 > self.max_seconds:
                print(f'Max time reached out: {t2}s! Break algorithm')
                break

            for i in range(x.shape[0]):
                if d_normalized[i] * (x[i, :].dot(w) + b) > 0:
                    n += 1
                    continue

                w += d_normalized[i] * x[i]
                b += d_normalized[i]
                n = 0

                self.iteration_count += 1

        self.coef_ = w
        self.intercept_ = b

        return w, b

