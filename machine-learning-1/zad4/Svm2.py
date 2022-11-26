import numpy as np
import cvxopt as cvx

from LinearClassifier import LinearClassifier


class Svm2(LinearClassifier):
    def __init__(self, c: float = 1.0, cls_lab=None):
        super().__init__(cls_lab)

        self.c_ = c
        self.svinds_ = None

    def fit(self, x: np.array, y: np.array):
        m, n = x.shape

        gp = -np.hstack((np.outer(y, np.ones(n + 1)) * np.hstack((x, np.ones((m, 1)))), -np.eye(m)))
        gpp = np.hstack((np.zeros((m, n + 1)), -np.eye(m)))
        g = cvx.matrix(np.vstack((gp, gpp)))

        h = cvx.matrix(np.concatenate((np.ones(m) * -1, np.zeros(m))))
        p = cvx.matrix(np.diag(np.concatenate((np.ones(n), np.zeros(m + 1)))))
        q = cvx.matrix(np.concatenate((np.zeros(n + 1), self.c_ * np.ones(m))))

        solution = cvx.solvers.qp(p, q, g, h)

        # print(g, h, p, q)
        # [print(k, ':', v) for k, v in solution.items()]

        self.coef_ = solution['x'][:n]
        self.intercept_ = solution['x'][n]
        self.svinds_ = np.nonzero(np.array(solution['z']) > 10e-5)[0]
