import numpy as np
from matplotlib import pyplot as plt

from LogisticRegressionL2 import LogisticRegressionL2
from MinimizeMethod import MinimizeMethod


def plot_class(clf: LogisticRegressionL2, x: np.array, y: np.array):
    x1min, x1max = np.min(x[:, 0]) - 0.1, np.max(x[:, 0]) + 0.1
    x2min, x2max = np.min(x[:, 1]) - 0.1, np.max(x[:, 1]) + 0.1

    x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 50), np.linspace(x2min, x2max, 50))
    z = clf.predict_proba(np.c_[x1.flatten(), x2.flatten()])[:, 1]
    print(z.shape)
    z = z.reshape(x1.shape)

    plt.contour(x1, x2, z, [0.25, 0.5, 0.75])
    plt.scatter(x[:, 0], x[:, 1], c=y + 1)
    plt.title(clf.__class__.__name__ + f'(method="{clf.method.name}")')


if __name__ == '__main__':
    plt.figure()
    n = 100
    m = 2

    x = np.vstack((np.random.randn(n, m) + 1, np.random.randn(n, m)))
    # x = np.hstack((x, np.random.randn(x.shape[0], 100)))
    y = np.vstack((-np.ones((n, 1)), np.ones((n,  1))))
    print(x.shape, y.shape)

    plt.scatter(x[:, 0], x[:, 1], c=y + 1, marker="o")
    # plt.show()

    clf1 = LogisticRegressionL2(method=MinimizeMethod.Newton_CG)
    clf1.fit(x, y.flatten())

    clf2 = LogisticRegressionL2(method=MinimizeMethod.L_BFGS_B)
    clf2.fit(x, y.flatten())

    plt.figure()
    print(clf1.coef_, clf1.intercept_)
    plot_class(clf1, x, y)

    plt.figure()
    print(clf2.coef_, clf2.intercept_)
    plot_class(clf2, x, y)

    plt.show()

"""
Zadanie:
1) napisać klasyfikator regresja logistyczna
2) dodać regularyzacje l2 
3) dodać wygładzoną regularyzacje l1

zad 2)
cost(w) + lambda * sum(wi**2) lub cost(w) + lambda * sum(f(w))
g = gradiend(w) + lambda 
q[1, :] += 2*lambda * w[1, :]
x.T * d * x
d = d + [[lambda, 0], [0, lambda]]

wygładzanie: soft_abs(w, a) = 1/a * log(cos_hiperboliczny(a*w))
diff_soft_abs(x, a) = tangens hiperboliczny(a * x)
"""