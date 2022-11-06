import numpy as np
from scipy import sparse

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
            'dirt_metric': DirtMetricEnum.INFO_GAIN,
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
        impurity_method = infogain if self.params['dirt_metric'] == DirtMetricEnum.INFO_GAIN else ginigain
        is_sparse = self.is_sparse(x) and self.is_sparse(y)
        print(type(x), type(y))

        impurity_values = [impurity_method(x[:, column] if is_sparse else x[column], y)
                           for column in (range(x.shape[1]) if is_sparse else x.columns)]
        # if is_sparse:
        #     impurity_values = []
        #     for column in range(x.shape[1]):
        #         # print(x[:, column])
        #         impurity_values.append(impurity_method(x[:, column], y))
        # else:
        #     impurity_values = [impurity_method(x[column], y) for column in x.columns]
        print(impurity_values)

        max_index = np.argmax(impurity_values)
        max_value = impurity_values[max_index]

        node.impurity_value = max_value
        node.id = max_index if is_sparse else x.columns[max_index]

        if max_value == 0 or len(impurity_values) == 1:
            return

        x_negatives, x_positives, y_negatives, y_positives = self.split(x, y, max_index)

        if (x_negatives.shape[0] if is_sparse else len(x_negatives)) > 0:
            left = Node()
            left.parent = node
            node.left = left
            self.build_tree(x_negatives, y_negatives, left)

        if (x_positives.shape[0] if is_sparse else len(x_positives)) > 0:
            right = Node()
            right.parent = node
            node.right = right
            self.build_tree(x_positives, y_positives, right)

    def cut_tree(self):
        pass

    @staticmethod
    def split(x, y, index):
        is_sparse = DecisionTree.is_sparse(x) and DecisionTree.is_sparse(y)
        column = x[:, index] if is_sparse else x[x.columns[index]]

        x_sparse_type = DecisionTree.get_sparse_type(x)
        x_negatives = x_sparse_type(x[column < 1]) if is_sparse else x[column < 1]
        x_positives = x_sparse_type(x[column > 0]) if is_sparse else x[column > 0]

        y_sparse_type = DecisionTree.get_sparse_type(y)
        y_negatives = y_sparse_type(y[column < 1]) if is_sparse else y[column < 1]
        y_positives = y_sparse_type(y[column > 0]) if is_sparse else y[column > 0]

        return x_negatives, x_positives, y_negatives, y_positives

    @staticmethod
    def is_sparse(column):
        return type(column) == sparse.csr_matrix or type(column) == sparse.csc_matrix

    @staticmethod
    def get_sparse_type(column):
        if DecisionTree.is_sparse(column) is False:
            raise 'The passed column is not sparse!'

        return sparse.csr_matrix if type(column) == sparse.csr_matrix else sparse.csc_matrix

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
