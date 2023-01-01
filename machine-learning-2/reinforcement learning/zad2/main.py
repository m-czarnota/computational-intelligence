from typing import Tuple
import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt


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

    action_count = 3
    action_prob = 1 / action_count

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
            sum(zrobie akcje -1 a on zrobi -x to nie ma wzmocnienia bo jego głupi ruch) dla x in range[-3, -1) - nieważne, czy zrobi ruch -2 czy -3, to jest jako jeden ruch, więc ilośc ruchów 1/2,
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

            actions[k] = -index - 1 if np.where(values[k] == value)[0].size != action_count else '*'
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
    gamma = 0.9
    win_prob = 18 / 37
    state_count = 100
    states = np.arange(state_count + 1)
    how_many_we_want_win = 100

    epsilon = 10e-6
    bellman_loop_count_iter = 100
    v = [Decimal(0) for _ in range(state_count + 1)]
    vv = [Decimal(0) for _ in range(state_count + 1)]

    values = np.empty(state_count + 1, dtype='object')
    best_actions = np.empty(state_count + 1, dtype='object')
    actions_by_state = np.empty(states.size, dtype='object')

    for state_iter, state in enumerate(states):
        actions_by_state[state_iter] = np.arange(1, np.min((state, state_count - state)) + 1)

    for _ in range(bellman_loop_count_iter):
        for state_iter, state in enumerate(states):
            v_for_state = []

            for action in actions_by_state[state_iter]:
                state_y_win = state + action
                state_y_lose = state - action

                enhancement_for_win = 0 if state_y_win != how_many_we_want_win else 1
                enhancement_for_lose = 0 if state_y_lose != how_many_we_want_win else 1

                win_val = Decimal(win_prob) * (Decimal(enhancement_for_win) + Decimal(gamma) * v[state_y_win])
                lose_val = Decimal(1 - win_prob) * (Decimal(enhancement_for_lose) + Decimal(gamma) * v[state_y_lose])

                v_for_state.append(win_val + lose_val)

            values[state_iter] = v_for_state

        for k in range(len(vv)):
            index, value = get_argmax_and_max_from_decimals(values[k])

            best_actions[k] = index + 1 if np.where(values[k] == value)[0].size != actions_by_state[k].size else 0
            vv[k] = value

        if get_max_absolute_val_from_decimals(subtract_two_decimals_lists(v, vv)) < epsilon:
            break

        v[:] = vv[:]

    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(len(v)), v)
    plt.title("Optimal value function V*(x)")
    plt.xlabel('State x (fund of player)')
    plt.ylabel('The best value function V*(x)')
    plt.savefig('optimal_value_function_v.png')
    # plt.show()

    plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(len(v)), best_actions)
    plt.title("Optimal strategy (0 means that your it doesn't matter - all have the same effect)")
    plt.xlabel('State x (fund of player)')
    plt.ylabel('The best action to do')
    plt.savefig('optimal_strategy.png')
    # plt.show()


def get_argmax_and_max_from_decimals(values: list) -> Tuple:
    index_max = 0
    val_max = Decimal(0)

    for val_iter, val in enumerate(values[1:]):
        if val > val_max:
            val_max = val
            index_max = val_iter

    return index_max, val_max


def subtract_two_decimals_lists(values1: list, values2: list) -> list:
    values = []
    for val1_iter, val1 in enumerate(values1):
        val2 = values2[val1_iter]
        values.append(val1 - val2)

    return values


def get_max_absolute_val_from_decimals(values: list) -> Decimal:
    values_abs = list(map(lambda val: abs(val), values))
    val_max = max(values_abs)

    return val_max


if __name__ == '__main__':
    zad4()
