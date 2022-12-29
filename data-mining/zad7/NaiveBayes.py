import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.class_labels = None
        self.probs_email_distribution = None
        self.condition_probs = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.class_labels = y.unique()
        self.probs_email_distribution = np.zeros((self.class_labels.size, y.size))
        self.condition_probs = np.zeros((self.class_labels.size, x.shape[1]))  # φk|y[0] = p(xj=k | y=0), φk|y[1] = p(xj=k | y=1)

        self.probs_email_distribution = self.get_email_probs_distribution(x, y)
        documents_by_labels = self.split_documents_by_label(x, y)

        for document_iter, document in x.iterrows():
            label = y[document_iter]

            for count_iter, (count_index, count) in enumerate(document.items()):
                self.condition_probs[label, count_iter] += 1 if count > 0 else 0

            for count_iter, _ in enumerate(document.items()):
                self.condition_probs[label, count_iter] /= documents_by_labels[label, count_iter]

        print(self.condition_probs)

    def predict(self, x: pd.DataFrame):
        return self.class_labels[np.argmax(self.predict_proba(x), axis=1)]

    def predict_proba(self, x: pd.DataFrame):
        scores = np.ones((x.shape[0], self.class_labels.size))

        for document_iter, document in x.iterrows():
            for label in self.class_labels:
                for count_iter, (count_index, count) in enumerate(document.items()):
                    value = self.probs_email_distribution[label, count_iter]
                    value = 10e-16 if value == 0 else value

                    scores[document_iter, label] += np.log2(value)

                scores[document_iter, label] += np.log2(self.condition_probs[label])

        return scores

    def get_email_probs_distribution(self, x: pd.DataFrame, y: pd.Series):
        """
        :return: φy = [0: p(y=0), 1: p(y=1)]
        """
        self.class_labels = y.unique() if self.class_labels is None else self.class_labels
        probs = np.zeros((self.class_labels.size, y.size))

        for class_label in self.class_labels:
            for label_iter, label in enumerate(y.items()):
                document = x.iloc[label_iter]
                probs[class_label, label_iter] = document.where(y == class_label).sum() / x.shape[0]

        return probs

    def split_documents_by_label(self, x: pd.DataFrame, y: pd.Series):
        """
        words count in each document by label [0, 1]:
            0: ...
            1: ...
        """
        self.class_labels = y.unique() if self.class_labels is None else self.class_labels
        separated_documents = np.zeros((self.class_labels.size, x.shape[1]))

        for label in self.class_labels:
            documents_by_label = x[y == label]
            documents_by_label = documents_by_label[documents_by_label > 0]
            documents_by_label = documents_by_label.apply(lambda val: pd.isna(val).sum(), axis=0)

            separated_documents[label, :] = documents_by_label[:]

        return separated_documents
