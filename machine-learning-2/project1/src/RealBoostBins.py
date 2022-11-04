import time
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class RealBoostBins(BaseEstimator, ClassifierMixin):
    OUTLIERS_RATIO = 0.05
    LOGIT_MAX = 2

    def __init__(self, t: int = 8, b: int = 8):
        self.T_ = t
        self.B_ = b
        self.mins_ = None
        self.maxes_ = None
        self.features_ = np.zeros(t, dtype='int16')  # indexes of selected features
        self.logits_ = np.zeros((t, b))  # b - number of bins (buckets)
        self.class_labels_ = None

    def fit(self, x: np.array, y: np.array):
        print("FIT...")
        t1 = time.time()

        self.class_labels_ = np.unique(y)  # pamiętamy słownik etykietek klas; zakładamy, że tu będą na pewno 2 klasy
        m, n = x.shape
        w = np.ones(m) / m  # każdy z przykładów będzie miał taką samą wagę na początku

        yy = np.ones(m)
        yy[y == self.class_labels_[0]] = -1.0

        self.set_mins_maxes(x)
        x_binned = self.calculate_bins(x)

        indexes_positives = yy == 1
        indexes_negatives = yy == -1

        indexer_positives = np.zeros((n, self.B_, m), dtype=np.bool)
        indexer_negatives = np.zeros((n, self.B_, m), dtype=np.bool)

        print('INDEXER START...')
        t1_indexer = time.time()
        for j in range(n):
            for b in range(self.B_):
                j_in_b = x_binned[:, j] == b
                indexer_positives[j, b] = np.logical_and(j_in_b, indexes_positives)
                indexer_negatives[j, b] = np.logical_and(j_in_b, indexes_negatives)
        t2_indexer = time.time()
        print(f'INDEXER DONE. [TIME: {t2_indexer - t1_indexer}s]')

        print('MAIN BOOSTING LOOP...')
        t1_main_loop = time.time()

        for t in range(self.T_):  # pętla po rundach boostingu
            err_exp_best = np.inf  # błąd wykładniczy najlepszy
            best_j = -1

            for j in range(n):  # pętla po cechach haara
                w_positives = np.zeros(self.B_)
                w_negatives = np.zeros(self.B_)
                logits_j = np.zeros(self.B_)

                for b in range(self.B_):
                    j_in_b = x_binned[:, j] == b  # where variable False is in bucket b
                    w_positives[b] = np.sum(w[indexer_positives[j, b]])
                    w_negatives[b] = np.sum(w[indexes_negatives[j, b]])

                    logits_j[b] = self.logit(w_positives[b], w_negatives[b])
                    # logits_j[b] = 0.0 if w_positives[b] == w_negatives[b] else (
                    #     -self.LOGIT_MAX if w_positives[b] == 0.0 else (
                    #         self.LOGIT_MAX if w_negatives[b] == 0.0 else np.clip(0.5 * np.log(w_positives[b] / w_negatives[b]), -self.LOGIT_MAX, self.LOGIT_MAX)
                    #     )
                    # )

                err_exp_j = np.sum(w * np.exp(-yy * logits_j[x_binned[:, j]]))
                if err_exp_j < err_exp_best:
                    err_exp_best = err_exp_j
                    best_j = j
                    self.logits_[t] = logits_j

            self.features_[t] = best_j
            w = w * np.exp(-yy * self.logits_[t, x_binned[:, best_j]]) / err_exp_best

        t2_main_loop = time.time()
        print(f'MAIN BOOSTING LOOP DONE. [TIME: {t2_main_loop - t1_main_loop}s]')

        t2 = time.time()
        print(f'FIT DONE. [TIME: {t2 - t1}s]')

    def decision_function(self, x):
        pass

    def predict(self, x):
        return self.class_labels_[1 * (self.decision_function(x) > 0.0)]

    def set_mins_maxes(self, x):
        m, n = x.shape

        print('FINDING RANGES...')
        t1_ranges = time.time()

        self.mins_ = np.zeros(n)
        self.maxes_ = np.zeros(n)

        for j in range(n):
            x_j_sorted = np.sort(x[:, j])
            self.mins_[j] = x_j_sorted[int(np.ceil(self.OUTLIERS_RATIO * m))]
            self.mins_[j] = x_j_sorted[int(np.floor((1.0 - self.OUTLIERS_RATIO) * m))]

        t2_ranges = time.time()
        print(f'FINDING RANGES DONE. [TIME: {t2_ranges - t1_ranges}s]')

    def calculate_bins(self, x):
        t1_binning = time.time()
        x_binned = np.clip(np.int8((x - self.mins_) / (self.maxes_ - self.mins_) * self.B_), 0, self.B_ - 1)
        t2_binning = time.time()
        print(f'BINNING DONE. [TIME: {t2_binning - t1_binning}s]')

        return x_binned

    def logit(self, w_positives, w_negatives):
        if w_positives == w_negatives:
            return 0.0

        if w_positives == 0.0:
            return -self.LOGIT_MAX

        if w_negatives == 0.0:
            return self.LOGIT_MAX

        return np.clip(0.5 * np.log(w_positives / w_negatives), -self.LOGIT_MAX, self.LOGIT_MAX)
