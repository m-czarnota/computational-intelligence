import graphviz
import numpy as np
from scipy import sparse

from Node import Node
from DirtMetricEnum import DirtMetricEnum
from Metrics import infogain, ginigain, entropy, freq


class DecisionTree:
    def __init__(self, params = {
            'depth': 10,
            'number_of_nodes': None,
            'threshold_value': None,
            'dirt_metric': DirtMetricEnum.INFO_GAIN,
        }):
        self.root = None
        self.params = params
        self.id_counter = 0

    def fit(self, x, y):
        self.root = Node()
        self.build_tree(x, y, self.root)

    def predict(self, x, y):
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
        # print(type(x), type(y))

        uniques_y, counts_y = freq(y, prob=False)
        uniques_y, probs_y = freq(y)
        impurity_values = [impurity_method(x[:, column] if is_sparse else x[column], y)
                           for column in (range(x.shape[1]) if is_sparse else x.columns)]
        # print(impurity_values)

        max_index = np.argmax(impurity_values)
        max_value = impurity_values[max_index]

        node.impurity_value = max_value
        node.id = f'{self.id_counter}'
        node.class_counts = counts_y
        node.class_probs = probs_y
        node.class_best = uniques_y[np.argmax(probs_y)]

        self.id_counter += 1

        # if max_value == 0 or len(impurity_values) == 1 or len(uniques_y) == 1 or self.pruning_codition(node):
        if max_value == 0 or len(impurity_values) == 1 or len(uniques_y) == 1 or node.depth() > self.params['depth']:
            node.column = None

            return
        else:
            node.column = x.columns[max_index]

        x_negatives, x_positives, y_negatives, y_positives = self.split(x, y, max_index)
        y_negatives_uniques = freq(y_negatives, prob=False)[0]
        y_positives_uniques = freq(y_positives, prob=False)[0]

        left = Node()
        left.parent = node
        node.left = left
        self.build_tree(x_negatives, y_negatives, left)

        right = Node()
        right.parent = node
        node.right = right
        self.build_tree(x_positives, y_positives, right)

    def cut_tree(self):
        pass

    @staticmethod
    def split(x, y, index):
        is_sparse = DecisionTree.is_sparse(x) and DecisionTree.is_sparse(y)

        if is_sparse:
            set1 = set(np.arange(x.shape[1]))
            set2 = set(x[:, index])
            intersect = list(set1.intersection(set2))
            print(intersect)

            return intersect

        column = x[x.columns[index]]

        x_negatives = x[column < 1]
        x_positives = x[column > 0]

        y_negatives = y[column < 1]
        y_positives = y[column > 0]

        return x_negatives, x_positives, y_negatives, y_positives

    @staticmethod
    def is_sparse(column):
        return type(column) == sparse.csr_matrix or type(column) == sparse.csc_matrix

    @property
    def tree_str(self):
        if self.root is None:
            return None

        dot = 'digraph G {'
        successors = [self.root]

        while len(successors) > 0:
            node = successors.pop(0)
            if node.column is not None:
                dot += f'{node.id} [label="{node.column}\n info gain: {node.impurity_value}\n counts: {node.class_counts}\n probs: {node.class_probs}\n class: {node.class_best}"]\n\n\n'
            else:
                dot += f'{node.id} [shape="box", label="Terminal\n info gain: {node.impurity_value}\n counts: {node.class_counts}\n probs: {node.class_probs}\n class: {node.class_best}"]\n\n\n'

            if node.left is not None:
                dot += f'{node.id}->{node.left.id} [label="False"]\n'
                successors.append(node.left)

            if node.right is not None:
                dot += f'{node.id}->{node.right.id} [label="True"]\n'
                successors.append(node.right)

        return dot + '}'

    @property
    def tree_var(self):
        if self.root is None:
            return None

        u = graphviz.Digraph('tree', filename='tree.png')
        successors = [self.root]

        while len(successors) > 0:
            node = successors.pop(0)

            if node.left is not None:
                u.edge(node.id, node.left.id)
                successors.append(node.left)

            if node.right is not None:
                u.edge(node.id, node.right.id)
                successors.append(node.right)

        return u

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
