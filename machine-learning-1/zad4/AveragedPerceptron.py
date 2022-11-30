import time
import numpy as np

from Perceptron import Perceptron


class AveragedPerceptron(Perceptron):
    def __init__(self, max_seconds: int = 5):
        super().__init__(max_seconds=max_seconds)

        self.c_ = None

    def fit(self, x, d):
        self.class_labels_ = np.unique(d)

        w = averaged_w = np.ones(x.shape[1])
        b = n = beta = 0
        c = 1
        t1 = time.time()

        while n < x.shape[0]:
            t2 = time.time() - t1
            if t2 > self.max_seconds:
                print(f'Max time reached out: {t2}s! Break algorithm')
                break

            for i in np.arange(x.shape[0]):
                if d[i] * (x[i, :].dot(w) + b) > 0:
                    n += 1
                    c += 1
                    continue

                actual_w = d[i] * x[i]

                w += actual_w
                b += d[i]
                n = 0

                averaged_w += actual_w * c  # accumulate the curr weight values (good weights reused)
                beta += d[i] * c

                self.iteration_count += 1

        """
        można robić wewnątrz pętli
        średnia = suma(xi) / n
        srednia(x[i+1]) = (x * i + x[i + 1]) / i + 1
        zmienić predycję!! prezentacja
        """

        self.coef_ = averaged_w
        self.intercept_ = b
        self.c_ = c

        return self.coef_, self.intercept_

    # def predict(self, x: np.array):
    #     results = np.sign(self.c_ * (x.dot(self.coef_) + self.intercept_))
    #     results_mapped = self.class_labels_[1 * (results > 0)] if self.class_labels_ is not None else results
    #
    #     return results_mapped
