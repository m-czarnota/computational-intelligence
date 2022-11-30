import time
import numpy as np

from LinearClassifier import LinearClassifier


class VotedPerceptron(LinearClassifier):
    def __init__(self):
        super().__init__()

        self.old_w_ = None
        self.old_b = None

    def fit(self, x, d):
        self.class_labels_ = np.unique(d)

        w, b = np.ones(x.shape[1]), 0
        old_weights = []
        old_b = []
        counter_life_weights = []
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

                old_weights.append(w)
                old_b.append(b)
                counter_life_weights.append(n)

                w += d[i] * x[i]
                b += d[i]
                n = 0

                self.iteration_count += 1

        self.coef_ = w
        self.intercept_ = b

        return w, b

    def predict(self, x: np.array):
        """
        tutaj dla każdego wektora z wagimi z listy old_weights zrobić predykcje
        predykcje * liczba iteracji ile przeżyło
        sum dla każdego wektora wag
        biere z tego znaki i mam odp
        """
        results = np.sign(self.c_ * np.sign(x.dot(self.coef_) + self.intercept_))
        results_mapped = self.class_labels_[1 * (results > 0)] if self.class_labels_ is not None else results

        return results_mapped
