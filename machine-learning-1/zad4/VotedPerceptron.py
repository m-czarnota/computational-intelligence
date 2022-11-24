import time
import numpy as np

from LinearClassifier import LinearClassifier


class VotedPerceptron(LinearClassifier):
    def __init__(self):
        super().__init__()

    def fit(self, x, d):
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
                if d[i] * (x[i, :].dot(w) + b) > 0:
                    n += 1
                    continue

                w += d[i] * x[i]
                b += d[i]
                n = 0

                self.iteration_count += 1

        self.coef_ = w
        self.intercept_ = b

        return w, b
