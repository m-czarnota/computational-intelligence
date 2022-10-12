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
    start = 2
    stop = 5
    step = 0.5
    samples = np.array([random.randint(0, int((stop - start) / step)) * step + start for i in range(100)])

    counts = np.unique(samples, return_counts=True)
    values = {counts[0][i]: counts[1][i] for i in range(len(counts[0]))}

    plt.figure()
    plt.bar(values.keys(), values.values(), width=0.3, color='g')

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
    plt.plot(x, y1, label='y1 = x^2 - 1')
    plt.plot(x, y0, label='y0 = -x - 1')
    plt.plot(x, y2, label='y2 = sin(x)')
    plt.legend(loc='upper left')

    minimum = np.minimum(y0, y2)
    minimum2 = np.minimum(y1, y0)

    plt.fill_between(x, minimum, minimum2, alpha=0.5)
    plt.show()


def zad5():
    start = -2
    stop = 2
    samples_count = 1000

    x = np.linspace(start, stop, samples_count)
    y = np.linspace(start, stop, samples_count)
    z = [0.5 if x_val + y_val < 0 else 1 / (1 + np.power(np.e, -(x_val + y_val))) for x_val, y_val in zip(x, y)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()


if __name__ == "__main__":
    zad5()
