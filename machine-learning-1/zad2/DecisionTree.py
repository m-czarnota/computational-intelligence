from Node import Node
from DirtMetricEnum import DirtMetricEnum


class DecisionTree:
    def __int__(self):
        self.root = None
        self.params = {
            'depth': None,
            'number_of_nodes': None,
            'threshold_value': None,
            'dirt_metric': DirtMetricEnum.CONDITION_ENTROPY,
        }

    def fit(self, x, y):
        root = Node()
        # self.rpart(x, y, root)

    def predict(self, x):
        pass

    def set_params(self, depth: int, number_of_nodes: int, threshold_value, dirt_metric: str = DirtMetricEnum.CONDITION_ENTROPY):
        pass

    def get_params(self):
        pass

    def cv_score(self):
        pass

    def build_tree(self, depth: int = None, number_of_nodes: int = None):
        pass

    def cut_tree(self):
        pass

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
    #             rpart(xl, yl, left)
    #             rpart(xr, yr, right)
    #     else:
    #         parent.left = None
    #         parent.right = None
    