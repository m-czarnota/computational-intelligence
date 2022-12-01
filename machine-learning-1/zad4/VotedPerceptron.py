import time
import numpy as np

from LinearClassifier import LinearClassifier


class VotedPerceptron(LinearClassifier):
    def __init__(self):
        super().__init__()

        self.old_w_ = None
        self.old_b_ = None
        self.counter_life_weights_ = None

    def fit(self, x, d):
        self.class_labels_ = np.unique(d)

        self.old_w_ = []
        self.old_b_ = []
        self.counter_life_weights_ = []

        w, b = np.ones(x.shape[1]), 0
        n = 0
        t1 = time.time()

        while n < x.shape[0]:
            t2 = time.time() - t1
            if t2 > self.max_seconds:
                print(f'Max time reached out: {t2}s! Break algorithm')
                break

            for i in np.arange(x.shape[0]):
                if d[i] * (x[i, :].dot(w) + b) > 0:
                    n += 1
                    continue

                self.old_w_.append(w)
                self.old_b_.append(b)
                self.counter_life_weights_.append(n)

                w += d[i] * x[i]
                b += d[i]
                n = 0

                self.iteration_count += 1

        self.coefs_ = w
        self.intercepts_ = b

        return w, b

    def predict(self, x: np.array):
        """
        tutaj dla każdego wektora z wagimi z listy old_weights zrobić predykcje
        predykcje * liczba iteracji ile przeżyło
        sum dla każdego wektora wag
        biere z tego znaki i mam odp
        """

        results = np.array([np.sign(x.dot(old_w) + old_b) for old_w, old_b in zip(self.old_w_, self.old_b_)])
        results = np.array([result * counter for result, counter in zip(results, self.counter_life_weights_)])
        results_mapped = np.array([np.sum(result) for result in results])
        results_mapped = np.sign(results_mapped)
        # results_mapped = [self.class_labels_[1 * (result > 0)] if self.class_labels_ is not None else result for result in results]
        # results_mapped = np.array([results * counter_life_weight for results, counter_life_weight in zip(results_mapped, self.counter_life_weights_)])

        return results_mapped

        # predictions = []
        #
        # for x_val in x:
        #     s = 0
        #
        #     for w, c in zip(self.old_w_, self.counter_life_weights_):
        #         s += c * np.sign(x_val.dot(w))
        #
        #     predictions.append(np.sign(s))
        #
        # return np.array(predictions)
