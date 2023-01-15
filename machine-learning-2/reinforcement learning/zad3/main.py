import numpy as np
import pandas as pd
import warnings


def zad1():
    """
    tablica Q, która reprezentuje funkcję wartości akcji (czyli sumę wszystkich przyszłych nagród):
        -1    -2
    0
    1
    2
    ..
    9   0.5   0.7
    10  nieważne jaki ruch, bo i tak przegramy
    użyć algorytmu QLearning
    po zakończeniu episodu resetujemy gre
    w każdym stanie wybranym zachłannie akcji powinno być optymalnie

    funkcje Q na początku wyzerować
    obserwuj aktualny stan x
    stan 9: która wartość największa da największą wartość Q
    najlepiej korzystać z epsilon zachłannie (epsilon powinien być mały)
    losuje p. jeśli p > epsilon, to wybierz akcje zachłanną, czyli wybieram akcje -2, bo jest większa wartość. inaczej wybierz akcje losowo

    skutkiem akcji dla -2 będzie 7, i jeszcze przeciwnik wykonuje ruch np. 1, więc jesteśmy w stanie 6
    i w tym stanie dokonujemy adaptacji funkcji Q dla stanu 9. znamy wzmocnienie dla 9, znamy następny stan, czyli 6

    r to wzmocnienie, które dostajemy po wykonaniu ruchu
    jak wygramy, to dostajemy wzmocnienie równe 1
    """
    gamma = 1
    eta = 0.2  # współczynnik szybkości uczenia
    epsilon = 0.5

    matches_count = 10
    q = pd.DataFrame(0, index=range(matches_count + 1), columns=[-1, -2])

    t = q.shape[0] - 1
    max_iter = 100000
    actual_iter = 0

    while True:
        # observe actual state t
        actual_q = q.loc[t]

        # select action
        actual_action_my = -(np.random.randint(2) if np.random.random() <= epsilon else np.argmax(actual_q)) - 1
        actual_action_opponent = -np.random.randint(2) - 1

        # do action
        next_state_after_my_turn = t + actual_action_my
        next_state_number = next_state_after_my_turn + actual_action_opponent
        if next_state_number < 0:
            next_state_number = 0
        actual_q = q.loc[next_state_number]

        # next state t + 1
        algorithm_next_state_number = next_state_number - 1
        if algorithm_next_state_number < 0:
            algorithm_next_state_number = np.random.randint(1, q.shape[0])
        next_q_after_actions = q.loc[algorithm_next_state_number]

        # r for t state
        r = 1 if next_state_number == 0 and next_state_after_my_turn > 0 else 0

        delta = r + gamma * next_q_after_actions.max() - actual_q.at[actual_action_my]
        q.at[algorithm_next_state_number, actual_action_my] = q.at[t, actual_action_my] + eta * delta

        t = algorithm_next_state_number

        actual_iter += 1
        if actual_iter >= max_iter:
            break

    print(q)
    print(q.idxmin(axis=1))


def zad3():
    """
    podobne dla zadania 1 na pierwszych laboratoriach
    przeszkody nie powinny być rozmieszczone losowo
    powinny one zajmować około 20% powierzchni
    zadaniem jest dotarcie do górnego lewego (wzmocnienie 0.5) albo górnego prawego narożnika (wzmocnienie 1)
    punkt startowy robota losowo, byle nie w przeszkodzie
    4 kierunki ruchu, rusza się do jednego albo drugiego narożnika

    funkcja Q w formie 3-wymiarowej tablicy: rozmiar 10x10x4
    można ją początkowo wyzerować
    w każdej komórce można wykonać ruch w 1 z 4 kierunków, więc to będzie ten 3-wymiar
    w wyniku uczenia ta tablica Q powinna się wypełnić takimi wartościami zachłannymi, aby prowadzić robota

    uczenie przerwać można po narzuconej liczbie iteracji, np. 1000
    im więcej episodów uczenia będzie, tym trasa powinna być lepsza
    """
    def select_action() -> str:
        state_actual = q.loc[t].copy()
        available_moves = []

        if t[0] > 0 and environment[t[0] - 1, t[1]] != -1:
            available_moves.append('up')
        if t[0] < environment_size - 1 and environment[t[0] + 1, t[1]] != -1:
            available_moves.append('down')
        if t[1] > 0 and environment[t[0], t[1] - 1] != -1:
            available_moves.append('left')
        if t[1] < environment_size - 1 and environment[t[0], t[1] + 1] != -1:
            available_moves.append('right')

        if np.random.random() <= epsilon:
            random_move = np.random.randint(len(available_moves))
            move = available_moves[random_move]
        else:
            moves_to_drop = set(state_actual.index).difference(set(available_moves))
            state_actual = state_actual.drop(moves_to_drop)
            move = state_actual.idxmax()

        return move

    def do_action(new_action: str) -> tuple:
        new_cords = np.copy(t)

        if new_action == 'up':
            new_cords[0] -= 1
        elif new_action == 'down':
            new_cords[0] += 1
        elif new_action == 'left':
            new_cords[1] -= 1
        elif new_action == 'right':
            new_cords[1] += 1

        return tuple(new_cords)

    def draw_new_cords() -> tuple:
        while True:
            cords = np.random.randint(0, environment_size, size=2)

            if environment[tuple(cords)] == 0:
                return tuple(cords)

    def map_direction_to_arrow(direction: str) -> str:
        if direction == 'up':
            return '↑'
        if direction == 'down':
            return '↓'
        if direction == 'left':
            return '←'
        if direction == 'right':
            return '→'

    environment_size = 10
    environment = np.full((environment_size, environment_size), 0, dtype='float')

    # set reinforcements
    environment[0, 0] = 0.5
    environment[0, environment_size - 1] = 1

    # set obstacles
    environment[2:6, 1] = -1
    environment[5, 2] = -1

    environment[6, 4:7] = -1
    environment[7, 6] = -1

    environment[2:4, 7:9] = -1
    environment[4, 5] = -1
    environment[5:8, 8] = -1

    environment[0:2, 4] = -1
    environment[1, 4:6] = -1
    environment[2, 5] = -1

    print(environment)
    print(f'obstacle size: {np.where(environment == -1)[0].size}')

    # learning params
    gamma = 1
    eta = 0.2  # współczynnik szybkości uczenia
    epsilon = 0.5

    indexes_without_obstacles = np.where(environment != -1)
    indexes = [(i, j) for i, j in zip(indexes_without_obstacles[0], indexes_without_obstacles[1])]
    q = pd.DataFrame(0, columns=['up', 'down', 'left', 'right'], index=pd.MultiIndex.from_tuples(indexes))
    t = (9, 5)

    max_iter = 100000
    actual_iter = 0

    while True:
        actual_state = q.loc[t]
        selected_action = select_action()
        new_cords = do_action(selected_action)

        r = environment[new_cords]
        if r == 0:
            next_state = q.loc[new_cords]
        else:
            new_cords = draw_new_cords()
            next_state = q.loc[new_cords]

        delta = r + gamma * next_state.max() - actual_state[selected_action]
        q.at[new_cords, selected_action] = actual_state[selected_action] + eta * delta

        t = new_cords

        actual_iter += 1
        if actual_iter >= max_iter:
            break

    environment = environment.astype('str')
    for multi_index, direction in q.idxmax(axis=1).items():
        environment[multi_index] = map_direction_to_arrow(direction)

    print(q)
    print(pd.DataFrame(environment))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        zad3()
