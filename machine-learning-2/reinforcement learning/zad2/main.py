import numpy as np


def zad1():
    gamma = 0.9
    r_1 = 1
    r_2 = 0.2
    epsilon = 10e-7

    v = np.zeros(5)
    vv = np.zeros(5)

    for i in range(100):
        vv[0] = np.max([gamma * v[1], -r_2 + gamma * v[2]])
        vv[1] = np.max([gamma * v[2], -r_2 + gamma * v[3]])
        vv[2] = np.max([gamma * v[3], r_1 + gamma * v[4]])
        vv[3] = np.max([r_1 + gamma * v[4], r_1 + gamma * v[4]])
        vv[4] = np.max([gamma * v[4], gamma * v[4]])

        if np.max(np.abs(v - vv)) < epsilon:
            break

        v[:] = vv[:]

    print(vv, v)

def zad2():
    """
    x = {0, ..., 10}, A = {-1, -2}
    if
    """

    gamma = 1
    epsilon = 10e-6

    v = np.zeros(11)
    vv = np.zeros(11)

    values = [0] * 11

    for i in range(100):
        values[10] = [0.5 * (gamma * v[7]) + 0.5 * (gamma * v[8]), 0.5 * (gamma * v[6]) + 0.5 * (gamma * v[7])]
        values[9] = [0.5 * (gamma * v[6]) + 0.5 * (gamma * v[7]), 0.5 * (gamma * v[5]) + 0.5 * (gamma * v[6])]
        values[8] = [0.5 * (gamma * v[5]) + 0.5 * (gamma * v[6]), 0.5 * (gamma * v[4]) + 0.5 * (gamma * v[5])]
        values[7] = [0.5 * (gamma * v[4]) + 0.5 * (gamma * v[5]), 0.5 * (gamma * v[3]) + 0.5 * (gamma * v[4])]
        values[6] = [0.5 * (gamma * v[3]) + 0.5 * (gamma * v[4]), 0.5 * (gamma * v[2]) + 0.5 * (gamma * v[3])]
        values[5] = [0.5 * (gamma * v[2]) + 0.5 * (gamma * v[3]), 0.5 * (gamma * v[1]) + 0.5 * (gamma * v[2])]
        values[4] = [0.5 * (gamma * v[1]) + 0.5 * (gamma * v[2]), 0.5 * (gamma * v[0]) + 0.5 * (gamma * v[1])]
        values[3] = [0.5 * (gamma * v[0]) + 0.5 * (gamma * v[1]), 1 * (1 + gamma * v[0])]
        values[2] = [1 * (1 + gamma * v[0]), 1 * (0 + gamma * v[0])]
        values[1] = [1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0])]
        values[0] = [1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0])]

        for k in range(vv.size):
            index = np.argmax(values[k])
            print(index)
            vv[k] = vv[values[k][index]]

        print(v)
        if np.max(np.abs(v - vv)) < epsilon:
            break

        for k in range(vv.size):
            v[k] = vv[k]

    print(v)


def zad4():
    """
    a = {1, ..., min(x, 100 - x)}
    równań belmana 101 ze 101 niewiadomymi
    pętla w pętli zrobić
    mogą być problemy numeryczne
    najwieksza możliwa dokładność
    """

if __name__ == '__main__':
    zad2()
