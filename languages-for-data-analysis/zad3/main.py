import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import cluster
import matplotlib.pyplot as plt


def zad1():
    def func(p):
        x, y, z, t = p
        return [
            10 - x * y,
            20 - y * z,
            30 - z * t,
            20 - x - y - z - t,
        ]

    solve = optimize.root(func, [1, 1, 1, 1])
    x, y, z, t = solve.x
    print(f'x={x}, y={y}, z={z}, t={t}')

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
        expression = 4 - x[0] ** 2 - x[1] ** 2
        result = -np.sqrt(expression) if expression > 0 else 0
        return result if reverse is False else -result

    cons = (
        {'type': 'ineq', 'fun': lambda x: 4 - x[0] ** 2 - x[1] ** 2}
    )

    solution = optimize.minimize(func, np.array([0.5, 0.5]), constraints=cons)
    print('minimum', solution.x)

    solution = optimize.minimize(func, np.array([0.5, 0.5]), args=[True], constraints=cons)
    print('maximum', solution.x)


def zad4():
    """
    wystarczy wyznaczyć istotną współrzędną środka ciężkości Xc, wyznaczene Yc jest trudne
    Xc = integrate(f(x) * x * dx, a, b) / integrate(f(x), a, b)
    można też integrate((f(x) - g(x)) * x * dx, a, b)

    wyznaczyć pkt przecięcia - brzeg dolny i górny
    """
    def f(x):
        return -x + 1

    def g(x):
        return x ** 2 - 1

    def h(x):
        return np.sin(x)

    start = 0
    stop = 1
    x = np.linspace(-2, 2, 1000)

    plt.figure()
    plt.plot(x, f(x))
    plt.plot(x, g(x))
    plt.plot(x, h(x))
    plt.show()

    xc = integrate.quad(f, start, stop)
    gc = integrate.quad(g, start, stop)
    hc = integrate.quad(h, start, stop)
    print(xc, gc, hc)


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
    zad4()
