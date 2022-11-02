from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class RealBoostBins(BaseEstimator, ClassifierMixin):
    def __init__(self, t: int = 8, b: int = 8):
        self.T_ = t
        self.logits_ = np.zeros((t, b))  # number of bins (buckets)
        self.class_labels_ = None

    def fit(self, x: np.array, y: np.array):
        self.class_labels_ = np.unique(y)  # pamiętamy słownik etykietek klas; zakładamy, że tu będą na pewno 2 klasy
        m, n = x.shape
        w = np.ones(m) / m  # każdy z przykładów będzie miał taką samą wagę na początku

        yy = np.ones(m)
        yy[y == self.class_labels_[0]] = -1.0

        for i in range(self.T_):  # pętla po rundach boostingu
            for j in range(n):  # pętla po cechach haara
                pass

    def decision_function(self, x):
        pass

    def predict(self, x):
        return self.class_labels_[1 * (self.decision_function(x) > 0.0)]
