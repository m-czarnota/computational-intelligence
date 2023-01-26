from scipy.optimize import minimize
from scipy.special import expit, log_expit
import numpy as np

from MinimizeMethod import MinimizeMethod
from PenaltyEnum import PenaltyEnum


class LogisticRegressionL2:
    def __init__(self, lambda_val: float = 1e-7, method: MinimizeMethod = MinimizeMethod.Newton_CG,
                 penalty: PenaltyEnum = PenaltyEnum.L2, **kwargs):
        self.lambda_val: float = lambda_val
        self.method: MinimizeMethod = method
        self.penalty: PenaltyEnum = penalty

        self.coef_ = np.array([])
        self.intercept_ = np.array([])

    def predict_proba(self, x: np.array) -> np.array:
        p = expit(x.dot(self.coef_) + self.intercept_)

        return np.array([1 - p, p]).T

    def predict(self, x: np.array):
        p = expit((x.dot(self.coef_) + self.intercept_))

        return 2 * (p > 0.5) - 1

    def fit(self, x: np.array, y: np.array) -> None:
        w0 = np.zeros(x.shape[1] + 1)
        res = minimize(self.cost, w0, args=(x, y), method=self.method.value, jac=self.grad, hess=self.hess,
                       options={"maxit": 100, "disp": True})
        print(res)

        self.coef_ = res['x'][1:]
        self.intercept_ = res['x'][0]

    def cost(self, w: np.array, x: np.array, y: np.array):
        return -np.sum(log_expit(y * (x.dot(w[1:]) + w[0])))
        # return -np.sum(log_expit(y * (x.dot(w[1:]) + w[0]))) + self.lambda_val * np.sum(w ** 2)  # l2 regularization

    @staticmethod
    def grad(w: np.array, x: np.array, y: np.array) -> np.array:
        v = y * expit(-y * (x.dot(w[1:]) + w[0]))

        return -np.hstack((np.ones((x.shape[0], 1)), x)).T.dot(v)

    @staticmethod
    def hess(w: np.array, x: np.array, y: np.array):
        p = expit(x.dot(w[1:]) + w[0])
        d = p * (1 - p)

        x_ = np.hstack((np.ones((x.shape[0], 1)), x))
        h = x_.T.dot(np.diag(d))
        h = h.dot(x_)

        return h

    @staticmethod
    def soft_abs(w: np.array, x: np.array, a: float = 1e3):
        t = a * x
        int1 = t > 100
        int2 = t < 100

        y = 1 / a * np.log(np.cosh(t))
        y[int1] = x * w[int1]
        y[int2] = -x * w[int2]

        return y
