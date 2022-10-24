from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt


def interpolate_knn(xi, x, y, k: int):
    tree = spatial.KDTree(x)
    res = tree.query(xi, k)

    interpolation = np.mean([y[index] for index in res[1]])

    return interpolation


def classify_knn(x, xc, yc, k):
    tree = spatial.KDTree(xc)
    res = tree.query(x, k)

    # print(res)
    labels = [yc[index] for index in res[1]]
    # print(labels)
    classification = np.unique(labels, return_counts=True)
    # print(classification)

    return classification[0][np.argmax(classification[1])]


def zad3(k, x, y):
    """
    k_min = 1, k_max <= len(x)
    wykluczyć pierwszy element, nauczyć sklasyfikować pozostałe
    potem powtórzyć dla wszystkich następnych
    k_error = ep / L
    L - liczba całego zbioru
    ep - liczba błędnie sklasyfikowanych próbek

    for k:
        for samples:
            dla każdego k wyznaczyć błąd

    szukać minimum
    """
    k_errors = []

    for k_val in k:
        errors = []

        for x_index, x_val in enumerate(x):
            point = x[x_index]
            other_points = np.vstack((x[0:x_index], x[x_index + 1:]))

            label = classify_knn(point, other_points, y, k_val)
            if label != y[x_index]:
                errors.append(x_val)

        k_errors.append(len(errors) / len(x))

    k_min_index = np.argmin(k_errors)
    k_min = np.min(k_errors)

    plt.figure()
    plt.plot(k, k_errors, label='k_errors')
    plt.scatter(k_min_index + 2, k_min, c='r', label='k min')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Xi = (0.5, 0.5)
    X = np.loadtxt('dane_2D_1_i.txt')
    Y = np.loadtxt('dane_2D_1_o.txt')
    K = 3

    # print(X)
    # print(np.vstack((X[2:4], X[6:8])))

    # interpolate_knn(Xi, X, Y, K)
    zad3(np.arange(2, 100), X, Y)
