from abc import ABC, abstractmethod


class LinearClassifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x, d):
        ...
