class Node:
    def __init__(self):
        self.id = None

        self.left = None
        self.right = None
        self.parent = None

        self.impurity_value = None

        self.best_attribute = None
        self.best_attribute_index = None

        self.classes = None
        self.classes_count = None
        self.classes_probs = None
        self.class_best = None

    def depth(self):
        depth = 1
        if self.parent is not None:
            depth += self.parent.depth()

        return depth

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_all_leafs(self, leafs: list = []):
        if self.is_leaf():
            leafs.append(self)
            return leafs

        if self.left is not None:
            return self.left.get_all_leafs(leafs)

        if self.right is not None:
            return self.right.get_all_leafs(leafs)

    def get_all_leafs_iter(self):
        successors = [self]
        leafs = []

        while len(successors) > 0:
            node = successors.pop(0)
            leafs.append(node)

            if node.left is not None:
                successors.append(node.left)

            if node.right is not None:
                successors.append(node.right)

        return successors
