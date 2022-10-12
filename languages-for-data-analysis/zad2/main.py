"""
zad 3 - polecenie bar
zad4 - narysować linie i zamalować obszar wspólny jako rozwiązanie
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def zad1():
    x = np.linspace(-np.pi, np.pi, 200)
    f = np.cos(x)
    g = np.sign(x)

    plt.figure()
    plt.plot(x, f, 'black', label='cos(x)')
    plt.plot(x, g, 'red', label='sign(x)')

    plt.grid()
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.yticks([-1, 0, 1])

    plt.legend(loc='upper left')
    plt.savefig('plot.png')
    plt.show()


def zad2():
    n = 1024
    squares_number = 9
    squares_number_sqrt = np.sqrt(squares_number)

    x = np.random.rand(n, 2)
    y = np.add(
        ((np.floor(x[:, 0] * squares_number_sqrt)) % squares_number_sqrt),
        ((np.floor(x[:, 1] * squares_number_sqrt)) % squares_number_sqrt)
    )
    y = y % 2
    colors = ['b' if val == 0 else 'r' for val in y]

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.show()


def zad3():
    """
    dokończyć, nie skończone!
    :return:
    """
    start = 2
    stop = 5
    step = 0.5
    samples = np.array([random.randint(0, int((stop - start) / step)) * step + start for i in range(100)])
    marks = np.linspace(start, stop, int((stop - start) / step))

    plt.figure()
    plt.bar(range(marks.size), marks)

    plt.xlabel('oceny')
    plt.ylabel('liczba ocen')

    plt.show()


def zad4():
    start = -2
    stop = 2
    samples_count = 1000

    x = np.linspace(start, stop, samples_count)
    y = np.linspace(start, stop, samples_count)

    y1 = y ** 2
    y1 -= 1
    y2 = np.sin(y)
    y0 = -x + 1

    plt.figure()
    plt.plot(x, y1)
    plt.plot(x, y0)
    plt.plot(x, y2)

    print(min((-x + 1) + [min(x)]))

    # where=y[y < min(y0 + [min(y1)])]
    plt.fill_between(y0, y1, y2)

    plt.show()


def zad5():
    start = -2
    stop = 2
    samples_count = 1000

    x = np.linspace(start, stop, samples_count)
    y = np.linspace(start, stop, samples_count)

    z = y.copy()
    z[x + y >= 0] = 1 / (1)


if __name__ == "__main__":
    zad4()
