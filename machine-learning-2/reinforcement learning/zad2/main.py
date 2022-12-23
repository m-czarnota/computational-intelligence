import numpy as np


def zad1():
    gamma = 0.9
    r_1 = 1
    r_2 = 0.2
    epsilon = 10e-7
    state_count = 5

    v = np.zeros(state_count)
    vv = np.zeros(state_count)

    actions = np.empty(state_count, dtype='object')
    values = np.empty(state_count, dtype='object')

    for i in range(100):
        values[0] = [gamma * v[1], -r_2 + gamma * v[2]]
        values[1] = [gamma * v[2], -r_2 + gamma * v[3]]
        values[2] = [gamma * v[3], r_1 + gamma * v[4]]
        values[3] = [r_1 + gamma * v[4], r_1 + gamma * v[4]]
        values[4] = [gamma * v[4], gamma * v[4]]

        for k in range(vv.size):
            index = np.argmax(values[k])
            value = values[k][index]

            actions[k] = index + 1 if np.where(values[k] == value)[0].size != 2 else '*'
            vv[k] = value

        if np.max(np.abs(v - vv)) < epsilon:
            break

        v[:] = vv[:]

    print(v[::-1])
    print(actions[::-1])


def zad2():
    """
    x = {0, ..., 10}, A = {-1, -2}
    if
    """

    gamma = 1
    epsilon = 10e-6
    state_count = 11

    v = np.zeros(state_count)
    vv = np.zeros(state_count)

    actions = np.empty(state_count, dtype='object')
    values = np.empty(state_count, dtype='object')

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
            value = values[k][index]

            actions[k] = -index - 1 if np.where(values[k] == value)[0].size != 2 else '*'
            vv[k] = value

        if np.max(np.abs(v - vv)) < epsilon:
            break

        v[:] = vv[:]

    print(v)
    print(actions)


def zad3():
    gamma = 1
    epsilon = 10e-6
    state_count = 11
    action_prob = 1 / 3

    v = np.zeros(state_count)
    vv = np.zeros(state_count)

    actions = np.empty(state_count, dtype='object')
    values = np.empty(state_count, dtype='object')

    for i in range(100):
        """
        jak to działa?
        values[10] = [1/ilosc_akcji * (gamma * v[ile_zapałek_zostaje]) + 1/ilosc_akcji * (gamma * v[ile_zapałek_zostaje]) + 1/ilosc_akcji * (gamma * v[ile_zapałek_zostaje])]
        values[10] = [prawdopodobieństwo że ja wykonam akcje -1 dla 10 zapałek a przeciwnik wykona akcje -3 i zostanie 6 zapałek + akcja moja -1 a typa -2 i zostanie 7 zapałek + .., ...]
        values[3] = [
            sum(zrobie akcje -1 a on zrobi -x to nie ma wzmocnienia bo jego głupi ruch) dla x in range(-3, -1) - nieważne, czy zrobi ruch -2 czy -3, to jest jako jeden ruch, więc ilośc ruchów 1/2,
            zrobie akcje -2 to nie ważne co on zrobi bo ja doprowadzam do wygranej i jest wzmocnienie,
            zrobie akcje -3 to przegrałem i nie ma więcej ruchów więc jest ruch 1
        ]
        """
        values[10] = [action_prob * (gamma * v[6]) + action_prob * (gamma * v[7]) + action_prob * (gamma * v[8]), action_prob * (gamma * v[5]) + action_prob * (gamma * v[6]) + action_prob * (gamma * v[7]), action_prob * (gamma * v[4]) + action_prob * (gamma * v[5]) + action_prob * (gamma * v[6])]
        values[9] = [action_prob * (gamma * v[5]) + action_prob * (gamma * v[6]) + action_prob * (gamma * v[7]), action_prob * (gamma * v[4]) + action_prob * (gamma * v[5]) + action_prob * (gamma * v[6]), action_prob * (gamma * v[3]) + action_prob * (gamma * v[4]) + action_prob * (gamma * v[5])]
        values[8] = [action_prob * (gamma * v[4]) + action_prob * (gamma * v[5]) + action_prob * (gamma * v[6]), action_prob * (gamma * v[3]) + action_prob * (gamma * v[4]) + action_prob * (gamma * v[5]), action_prob * (gamma * v[2]) + action_prob * (gamma * v[3]) + action_prob * (gamma * v[4])]
        values[7] = [action_prob * (gamma * v[3]) + action_prob * (gamma * v[4]) + action_prob * (gamma * v[5]), action_prob * (gamma * v[2]) + action_prob * (gamma * v[3]) + action_prob * (gamma * v[4]), action_prob * (gamma * v[1]) + action_prob * (gamma * v[2]) + action_prob * (gamma * v[3])]
        values[6] = [action_prob * (gamma * v[2]) + action_prob * (gamma * v[3]) + action_prob * (gamma * v[4]), action_prob * (gamma * v[1]) + action_prob * (gamma * v[2]) + action_prob * (gamma * v[3]), action_prob * (gamma * v[0]) + action_prob * (gamma * v[1]) + action_prob * (gamma * v[2])]
        values[5] = [action_prob * (gamma * v[1]) + action_prob * (gamma * v[2]) + action_prob * (gamma * v[3]), action_prob * (gamma * v[0]) + action_prob * (gamma * v[1]) + action_prob * (gamma * v[2])]
        values[4] = [action_prob * (gamma * v[0]) + action_prob * (gamma * v[1]) + action_prob * (gamma * v[2]), 0.5 * (gamma * v[0]) + 0.5 * (gamma * v[0]), 1 * (1 + gamma * v[0])]
        values[3] = [0.5 * (0 + gamma * v[0]) + 0.5 * (gamma * v[1]), 1 * (1 + gamma * v[0]), 1 * (0 + gamma * v[0])]
        values[2] = [1 * (1 + gamma * v[0]), 1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0])]
        values[1] = [1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0])]
        values[0] = [1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0]), 1 * (0 + gamma * v[0])]

        for k in range(vv.size):
            index = np.argmax(values[k])
            value = values[k][index]

            actions[k] = -index - 1 if np.where(values[k] == value)[0].size != 2 else '*'
            vv[k] = value

        if np.max(np.abs(v - vv)) < epsilon:
            break

        v[:] = vv[:]

    print(v[::-1])
    print(actions[::-1])


def zad4():
    """
    a = {1, ..., min(x, 100 - x)}
    równań belmana 101 ze 101 niewiadomymi
    pętla w pętli zrobić
    mogą być problemy numeryczne
    najwieksza możliwa dokładność
    """


if __name__ == '__main__':
    zad3()
