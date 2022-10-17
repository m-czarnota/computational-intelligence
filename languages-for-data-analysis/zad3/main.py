import math
import numpy as np
from scipy import integrate
from scipy import optimize
import numpy as np
from scipy import cluster
import matplotlib.pyplot as plt


def zad1():
    def func(t, y):
        return [
            10 - y[0] * y[1],
            20 - y[1] * y[2],
            30 - y[2] * y[3],
            20 - y[0] - y[1] - y[2] - y[3],
        ]

    y0 = [1, -5]
    result = integrate.solve_ivp(func, [0, 5], y0)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(result.t, result.y[0])

    plt.subplot(2, 2, 2)
    plt.plot(result.t, result.y[1])

    plt.show()


def zad2():
    def func(x, reverse: bool = False):
        result = x[0] ** 2 + x[1] ** 2 - x[0] * x[1] + x[0] + x[1]
        return result if reverse is False else -result

    solution = optimize.minimize(func, np.array([0, 0]), bounds=((-3, None), (-3, None)))
    # print(solution)
    print('minimum', solution.x)

    solution = optimize.minimize(func, np.array([0, 0]), args=[True], bounds=((-3, 0), (-3, 0)))
    print('maximum', solution.x)


def zad3():
    def func(x, reverse: bool = False):
        result = -math.sqrt(4 - x[0] ** 2 - x[1] ** 2)
        return result if reverse is False else -result

    cons = (
        {'type': 'ineq', 'fun': lambda x: 4 - x[0] ** 2 - x[1] ** 2}
    )

    solution = optimize.minimize(func, np.array([1, 1]), constraints=cons)
    print('minimum', solution.x)

    solution = optimize.minimize(func, np.array([0, 0]), args=[True], constraints=cons)
    print('maximum', solution.x)


def zad4():
    """
    wystarczy wyznaczyć istotną współrzędną środka ciężkości Xc, wyznaczene Yc jest trudne
    Xc = integrate(f(x) * x * dx, a, b) / integrate(f(x), a, b)
    można też integrate((f(x) - g(x)) * x * dx, a, b)

    wyznaczyć pkt przecięcia - brzeg dolny i górny
    """
    pass


def zad5():
    """
    jedna z technik to wyznaczenie krzywej obrazującej jakość klasteryzacji od liczby klastrów
    jak mam 100 punktów i 100 klastrów i każdy z klastrów był w innym pkt, to wartość d z kmeans byłaby równa 0
    znaleźć miejsce, gdzie krzywa zaczyna się wypłaszczać w miejscu x
    metodą wzrokową wyłapać
    w pętli puścić klasteryzacje od 1 do tyle ile jest próbek, dla każdej klasteryzacji złapać d, zrobić wykres i wzrokowo określić
    """
    filenames = ['dane_2D.txt', 'dane_3D.txt']
    for filename in filenames:
        M = np.loadtxt(filename)
        d_dict = {}

        for i in range(1, len(M) + 1):
            c, d = cluster.vq.kmeans(M, i)
            d_dict[i] = d

        plt.figure()
        plt.scatter(d_dict.keys(), d_dict.values())
        plt.title(f'plot for {filename}')
        plt.show()

    # krzywa dla danych 2D zaczyna się wypłaszczać w 10 próbce
    # krzywa dla danych 3D zaczyna się wypłaszczać w 13 próbce


if __name__ == '__main__':
    zad1()
