import numpy as np
import cvxopt as cvx

from LinearClassifier import LinearClassifier


class Svm1(LinearClassifier):
    def __init__(self, cls_lab=None):
        super().__init__(cls_lab)

        self.svinds_ = None

    def fit(self, x: np.array, y: np.array):
        m, n = x.shape

        g = cvx.matrix(np.outer(y, np.ones(n + 1)) * np.hstack((x, np.ones((m, 1)))))
        h = cvx.matrix(np.ones(m) * -1)
        q = cvx.matrix(np.zeros(n + 1))

        p = np.eye(n + 1)
        p[-1, -1] = 0
        p = cvx.matrix(p)

        solution = cvx.solvers.qp(p, q, g, h)

        # print(g, h, p, q)
        # [print(k, ':', v) for k, v in solution.items()]

        self.coef_ = solution['x'][:n]
        self.intercept_ = solution['x'][n]
        self.svinds_ = np.zeros(np.array(solution['z']) > 10e-5)[0]

