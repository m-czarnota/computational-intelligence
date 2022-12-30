import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_labels = None
        self.apriori_labels = None
        self.condition_probs_distributions = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.class_labels = y.unique()

        self.apriori_labels = self.get_apriori_labels(y)
        self.condition_probs_distributions = self.get_condition_probs_distributions(x, y)

        print(self.condition_probs_distributions)

    def predict(self, x: pd.DataFrame):
        return self.class_labels[np.argmax(self.predict_proba(x), axis=1)]

    def predict_proba(self, x: pd.DataFrame):
        scores = np.ones((x.shape[0], self.class_labels.size))

        for document_iter, (document_index, document) in enumerate(x.iterrows()):
            for label in self.class_labels:
                for word_iter, _ in enumerate(document.items()):
                    value = self.condition_probs_distributions[label, word_iter]
                    value = 1e-16 if value == 0 else value

                    scores[document_iter, label] += np.log(value)

                scores[document_iter, label] += np.log(self.apriori_labels[label])

        return scores

    def get_apriori_labels(self, y: pd.Series):
        """
        Calculates how often class occurs in decisions vector (in our default case 1/2 and 1/2).
        """
        apriori_labels = np.zeros(self.class_labels.size)

        for label_iter, label in enumerate(self.class_labels):
            apriori_labels[label_iter] = np.sum(y == label) / y.size

        return apriori_labels

    def get_condition_probs_distributions(self, x: pd.DataFrame, y: pd.Series):
        """
        Calculates condition probabilities distributions.
        Frequency calc explanation example:
            * when label == 1 and word == 543 then +1 in [label, word]
        Probs calc explanation example:
            * divide each counts in specific label by sum of words count from all documents
            * before dividing add +1 to counts as laplace fix
        """
        condition_distributions = np.zeros((self.class_labels.size, x.shape[1]))

        for document_index, document in x.iterrows():
            label = y[document_index]

            for word_iter, (word, count) in enumerate(document.items()):
                condition_distributions[label, word_iter] += 1 if count > 0 else 0

        for label in self.class_labels:
            label_count = self.apriori_labels[label] * x.shape[0]
            word_count = x[y == label].sum(axis=0).to_numpy()
            # word_count = x[y == label].applymap(lambda val: 1 if val > 0 else val).sum(axis=0).sum()

            condition_distributions[label] = (condition_distributions[label] + 1) / (word_count + x.shape[1])

        return condition_distributions
