from LinearClassifier import LinearClassifier
import time


class Perceptron(LinearClassifier):
    def __init__(self, gamma: float = 0.0, max_seconds: int = 3600, **kwargs):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.max_seconds = max_seconds

    def fit(self, x, d):
        w, b = [0, 0], 0
        n = 0
        t1 = time.time()

        while n < x.shape[0] and time.time() - t1 < self.max_seconds:
            for i in range(x.shape[0]):
                if d[i] * (x[i, :].dot(w) + b) > 0:
                    n += 1
                    continue

                w += d[i] * x[i]
                b += d[i]
                n = 0

        return w, b

