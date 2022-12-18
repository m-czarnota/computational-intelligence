import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.class_labels = None
        self.probs_y = None
        self.condition_probs = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.class_labels = y.unique()
        self.probs_y = np.zeros(y.shape)
        self.condition_probs = np.empty((self.class_labels.size, x.shape[1]), dtype='object')

        for label_iter, label in enumerate(y.items()):
            document = x.iloc[label_iter]
            self.probs_y[label_iter] = document.where(y == 1).sum() / x.shape[0]

        print(self.probs_y)

    @staticmethod
    def calc_stats(x: pd.DataFrame, y: pd.Series):
        mean = x.groupby(y).apply(np.mean).to_numpy()
        # var =
