# xi, di
# xi - współrzędne x
# di - decyzje
# <w, x> to wektor
# <w, x> + b
#
# def perceptron(x, d):
#     n == lr
#     stop_condition = n < count_samples
#     while (not stop_condition):
#         mozna dodać warunek bezpieczeństwa na czas
#
#         iterowanie po próbkach i
#
#         if (d[i] * iloczyn_skalarny(w, x) + b <= 0):
#             #update
#             w = w + d[i] * x[i]
#             b = b + d[i] * 1
#             n = 0
#         else:
#             n += 1
#
#         return w, b
#
# są próbki dodatnie, są próbki ujemne. chcemy znaleźć prostą która separuje próbki

# def dane_liniowo_separowalne():
#     w, b = [1, 3], -1
#     x = np.random.rand(10, 2) * 3
#     d = np.sign(x.dot(w) + b)
#     x.dot(w) - iloczyn skalarny <w, x>
#
# w i b to parametry prostej (hiperpłaszczyzny - równanie hiperpłaszczyzny: <w, x> + b = 0, czyli sum(w[i] * x[i] + b) = 0)
# jak znaleźć b? mając pkt (0,0) liczymy x/||w||
#
# aby narysować prostą, trzeba wyznaczyć dwa pkt

# def funkcja_uniwersalna_brzydka():
#     contour
#     [x1, x2] = meshgrid(zakres)  # daje nam 2 macierze współrzędnych
#     X = np.array([x1.flatten(), x2.flatten()]).T  # flatten robi kopię, raven modyfikuje oryginalne
#
#     z = clf.predict(X)
#     z = z.reshape((n, n))
#     contour(x1, x2, z, [0, 0])


import numpy as np
import matplotlib.pyplot as plt

from LinearClassifier import LinearClassifier
from Perceptron import Perceptron


def linear_separetly_dataset():
    w, b = [1, 1], -1
    x = np.random.randn(100, 2)
    d = np.sign(x.dot(w) + b)

    return x, d


def plot_class(x: np.array, y: np.array, clf: LinearClassifier):
    n, m = x.shape

    [x1, x2] = np.meshgrid(n, m)
    x_1_2_flatten = np.array([x1.flatten(), x2.flatten()]).T

    z = clf.predict(x_1_2_flatten)
    z = z.reshape((n, n))

    plt.contour(x1, x2, z, [0, 0])
    plt.show()


if __name__ == '__main__':
    x, d = linear_separetly_dataset()

    perceptron = Perceptron()
    w, b = perceptron.fit(x, d)
    print(w, b)

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=d)
    plt.show()

    # plot_class(x, d, perceptron)


