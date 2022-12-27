import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.class_labels = None
        self.probs_y = None
        self.condition_probs = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.class_labels = y.unique()
        self.probs_y = np.zeros((self.class_labels.size, y.shape))  # φy[0] = p(y=0), φy[1] = p(y=1)
        self.condition_probs = np.zeros((self.class_labels.size, x.shape[1]))  # φk|y[0] = p(xj=k | y=0), φk|y[1] = p(xj=k | y=1)

        for class_label in self.class_labels:
            for label_iter, label in enumerate(y.items()):
                document = x.iloc[label_iter]
                self.probs_y[class_label, label_iter] = document.where(y == class_label).sum() / x.shape[0]

        print(self.probs_y)

        documents_by_labels = np.zeros((self.class_labels.size, x.shape[1]))  # words count in each document by label [0, 1]
        for label in self.class_labels:
            documents_by_label = x[y == label]
            documents_by_label = documents_by_label[documents_by_label > 0]
            documents_by_label = documents_by_label.apply(lambda val: pd.isna(val).sum(), axis=0)

            documents_by_labels[label, :] = documents_by_label[:]

        for document_iter, document in x.iterrows():
            label = y[document_iter]

            for count_iter, (count_index, count) in enumerate(document.items()):
                self.condition_probs[label, count_iter] += 1 if count > 0 else 0
                self.condition_probs[label, count_iter] /= documents_by_labels[label, count_iter]

        print(self.condition_probs)

    @staticmethod
    def calc_stats(x: pd.DataFrame, y: pd.Series):
        mean = x.groupby(y).apply(np.mean).to_numpy()
        # var =
