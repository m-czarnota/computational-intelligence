import numpy as np
import pandas as pd


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
    epsilon = 0.04

    matches_count = 10
    q = pd.DataFrame(0, index=range(matches_count + 1), columns=[-1, -2])

    t = q.shape[0] - 1
    max_iter = 1000
    actual_iter = 0

    while True:
        actual_q = q.loc[t]
        actual_action_my = -(np.random.randint(1) if np.random.random() <= epsilon else np.argmax(actual_q)) - 1
        actual_action_opponent = -np.random.randint(1) - 1

        next_state_after_my_turn = t + actual_action_my
        next_state_number = next_state_after_my_turn + actual_action_opponent
        if next_state_number < 0:
            next_state_number = 0

        next_q_after_actions = q.loc[next_state_number]

        r = 1 if next_state_number == 0 and next_state_after_my_turn > 0 else 0
        delta = r + gamma * np.max(next_q_after_actions) - actual_q.at[actual_action_my]
        q.at[t - 1, actual_action_my] = q.at[t, actual_action_my] + eta * delta

        t = np.random.randint(1, q.shape[0] - 1) if t <= 0 else t - 1  # if t is 1 then -1 will give you 0

        actual_iter += 1
        if actual_iter >= max_iter:
            break

    print(q)


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


if __name__ == '__main__':
    zad1()
