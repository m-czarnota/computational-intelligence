import numpy as np
from scipy import sparse

from DecisionTreeParams import DecisionTreeParams
from Node import Node
from DirtMetricEnum import DirtMetricEnum
from Metrics import infogain, ginigain, freq


class DecisionTree:
    def __init__(self, params: DecisionTreeParams = DecisionTreeParams()):
        self.root = None
        self.params: DecisionTreeParams = params
        self.id_counter: int = 0

    def fit(self, x, y):
        self.root = Node()
        self.build_tree(x, y, self.root)

    def predict(self, x):
        y = []

        for i in range(x.shape[0]):
            node = self.root

            while True:
                if node.is_leaf():
                    y.append(node.class_best)
                    break

                attribute_value = np.bool(x.iloc[i, node.best_attribute_index])

                if attribute_value is False:
                    node = node.left
                    continue

                if attribute_value is True:
                    node = node.right
                    continue

        return y

    def set_params(self, params: DecisionTreeParams):
        self.params = params

    def get_params(self):
        return self.params

    def cv_score(self):
        pass

    def build_tree(self, x, y, node: Node):
        impurity_method = infogain if self.params.dirt_metric == DirtMetricEnum.INFO_GAIN else ginigain
        is_sparse = self.is_sparse(x) and self.is_sparse(y)

        uniques_y, probs_y = freq(y)
        impurity_values = [impurity_method(x[:, column] if is_sparse else x[column], y)
                           for column in (range(x.shape[1]) if is_sparse else x.columns)]

        max_index = np.argmax(impurity_values)
        max_value = impurity_values[max_index]

        node.impurity_value = max_value
        node.id = f'{self.id_counter}'
        node.classes, node.classes_count = freq(y, prob=False)
        node.classes_probs = probs_y
        node.class_best = uniques_y[np.argmax(probs_y)]

        self.id_counter += 1

        if max_value == 0 or len(impurity_values) == 1 or len(uniques_y) == 1 or self.pruning_conditions(node):
            return

        node.best_attribute = x.columns[max_index]
        node.best_attribute_index = max_index
        x_negatives, x_positives, y_negatives, y_positives = self.split(x, y, max_index)

        left = Node()
        left.parent = node
        node.left = left
        self.build_tree(x_negatives, y_negatives, left)

        right = Node()
        right.parent = node
        node.right = right
        self.build_tree(x_positives, y_positives, right)

    def pruning_conditions(self, node) -> bool:
        if node.depth() > self.params.depth:
            return True

        return False

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
    def tree_(self):
        if self.root is None:
            return None

        dot = 'digraph G {'
        successors = [self.root]

        while len(successors) > 0:
            node = successors.pop(0)

            shape = 'box' if node.best_attribute is None else 'ellipse'
            label = f'info gain: {node.impurity_value}\n classes: {node.classes}\n counts: {node.classes_count}\n probs: {node.classes_probs}\n best class: {node.class_best}'
            label = f'{node.best_attribute if node.best_attribute is not None else "Terminal"}\n {label}'
            dot += f'{node.id} [shape="{shape}", label="{label}"]'

            if node.left is not None:
                dot += f'{node.id}->{node.left.id} [label="False"]\n'
                successors.append(node.left)

            if node.right is not None:
                dot += f'{node.id}->{node.right.id} [label="True"]\n'
                successors.append(node.right)

        return dot + '}'
