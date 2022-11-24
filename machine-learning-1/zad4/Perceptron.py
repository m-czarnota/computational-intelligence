import numpy as np

from LinearClassifier import LinearClassifier
import time


class Perceptron(LinearClassifier):
    def __init__(self, gamma: float = 0.0, max_seconds: int = 5, **kwargs):
        super().__init__(max_seconds=max_seconds, **kwargs)

        self.gamma = gamma
        self.max_seconds = max_seconds

        self.iteration_count = 0

    def fit(self, x: np.array, d: np.array):
        self.class_labels_ = np.unique(d)

        w, b = np.ones(x.shape[1]), 0
        n = 0
        t1 = time.time()

        while n < x.shape[0]:
            t2 = time.time() - t1
            if t2 > self.max_seconds:
                print(f'Max time reached out: {t2}s! Break algorithm')
                break

            for i in np.arange(x.shape[0]):
                # pkt 3, modyfikacja funkcji fit. gamma * ||w||
                if d[i] * (x[i, :].dot(w) + b) > self.gamma * np.linalg.norm(w):
                    n += 1
                    continue

                w += d[i] * x[i]
                b += d[i]
                n = 0

                self.iteration_count += 1

        self.coef_ = w
        self.intercept_ = b

        return w, b

