import time
import numpy as np

from LinearClassifier import LinearClassifier


class AveragedPerceptron(LinearClassifier):
    def __init__(self):
        super().__init__()

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

                w += d[i] * x[i]
                averaged_w += d[i] * c * x[i]  # accumulate the curr weight values (good weights reused)
                b += d[i]
                beta += d[i] * c
                n = 0

                self.iteration_count += 1

        """
        można robić wewnątrz pętli
        średnia = suma(xi) / n
        srednia(x[i+1]) = (x * i + x[i + 1]) / i + 1
        zmienić predycję!! prezentacja
        """

        self.coef_ = w - 1 / c * averaged_w
        self.intercept_ = b - 1 / c * beta
        self.c_ = c

        return self.coef_, self.intercept_

    # def predict(self, x: np.array):
    #     results = np.sign(self.c_ * (x.dot(self.coef_) + self.intercept_))
    #     results_mapped = self.class_labels_[1 * (results > 0)] if self.class_labels_ is not None else results
    #
    #     return results_mapped
