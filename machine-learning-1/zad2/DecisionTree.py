import numpy as np

from Node import Node
from DirtMetricEnum import DirtMetricEnum
from Metrics import infogain, ginigain, entropy, freq


class DecisionTree:
    def __init__(self):
        self.root = None
        self.params = {
            'depth': None,
            'number_of_nodes': None,
            'threshold_value': None,
            'dirt_metric': DirtMetricEnum.CONDITION_ENTROPY,
        }

    def fit(self, x, y):
        self.root = Node()
        self.build_tree(x, y, self.root)

    def predict(self, x):
        pass

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_params(self):
        return self.params

    def cv_score(self):
        pass

    def build_tree(self, x, y, node: Node):
        impurity_method = entropy if self.params['dirt_metrics'] == DirtMetricEnum.CONDITION_ENTROPY else ginigain
        impurity_values = [impurity_method(x[column], y) for column in x.columns]

        max_index = np.argmax(impurity_values)
        max_value = impurity_values[max_index]

        node.impurity_value = max_value
        node.column = x.columns[max_index]

        if max_value == 0 or len(impurity_values) == 1:
            return

        positives, negatives = self.split(x, max_index)

        if len(negatives) > 0:
            left = Node()
            left.parent = node
            node.left = left
            self.build_tree(negatives, y, left)

        if len(positives) > 0:
            right = Node()
            right.parent = node
            node.right = right
            self.build_tree(positives, y, right)

    def cut_tree(self):
        pass

    def split(self, x, index):
        column = x[x.columns[index]]
        positives = x[column > 0].drop([x.columns[index]], axis=1)
        negatives = x[column < 1].drop([x.columns[index]], axis=1)

        return positives, negatives

    # def rpart(self, x, y, node):
    #     xi, pi = freq(y)
    #     node.pi = pi
    #     node.xi = xi
    #
    #     for i in range(x.shape[1]):
    #         ig[i] = 'atrybuty'
    #
    #     ini = np.argmax(iy)
    #     if ig[id] > 0:
    #         node = Node()
    #         xl, yl, xr, yr = split(x, y, x[:, ind])
    #         rpart()
    #
    #         if (podzial):
    #             xl, yl, xr, yr, split(x, y, ind)
    #
    #             left = Node()
    #             left.parent = node
    #
    #             right = Node()
    #             right.parent = node
    #
    #             node.left = left
    #             node.right = right
    #
    #             rpart(xl, yl, left)
    #             rpart(xr, yr, right)
    #     else:
    #         parent.left = None
    #         parent.right = None
