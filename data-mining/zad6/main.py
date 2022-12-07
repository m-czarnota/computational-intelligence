"""
(1-d) to chęć odwiedzenia węzła
każdy z węzłów może być odwiedzony z takim samym prawdopodobieństwem, czyli 1/4
uzyskuje się wartości pagerank po to, aby wykorzystać go do wyszukiwarki stron, określa kolejność przy wyszukiwaniu
nie da się obliczyć tego w jednej linijce, bo to jet łańcuch markova, przejście przez stany

podać wartość page rank dla węzła a i e. przekształcić wzór. wykazać jaka to jest zależność do użycia macierzowego, a nie pojedynczego b
można wykazać analitycznie, albo eksperymentalnie szukając dzięki kodowaniu
skoro a i e mają podobną strukturę, to nieważne jake d będzie przyjęte, wartości page ranks będą takie same
uzależnić wartość p od d
można jako komentarz. łatwiej jest to zrobić na kartce. rozpisać sobie wzór page rank dla węzła i zobaczyć jakie d wychodzi

w page rank warunek zatrzymania może być iteracyjny, różnica
"""
import numpy as np


def page_rank_algorithm(a: np.array, d: float = 0.75, iteration_count: int = np.inf):
    n, m = a.shape
    epsilon = 0.01

    e = np.ones(a.shape)
    val_0 = e[:, 0] / n
    k = 1

    while True:
        val_k = val_0.dot((1 - d) * (e / n) + d * a.T)

        if np.linalg.norm(val_k - val_0) < epsilon or k >= iteration_count:
            val_0 = val_k
            break

        val_0 = val_k
        k += 1

    return val_0


def zad2():
    n, m = a.shape


if __name__ == '__main__':
    a = np.array([
        [0, 1, 1, 1],
        [1 / 3, 0, 0, 0],
        [1 / 3, 0, 0, 0],
        [1 / 3, 0, 0, 0],
    ])
    print(page_rank_algorithm(a, iteration_count=1))

    a = np.array([
        [0, 1/3, 1/4, 1/3, 0],
        [0, 0, 1/4, 0, 0],
        [0, 1/3, 0, 1/3, 0],
        [0, 0, 1/4, 0, 0],
        [0, 1/3, 1/4, 1/3, 0],
    ])
    print(page_rank_algorithm(a, 0.9))

"""
zad 1:
    [0.625 0.125 0.125 0.125]
zad 2:
    1: 
        PR(A) = (1-d) + d * ((PR(B)/1 + PR(C)/1 + PR(D)/1) / 3)
        PR(E) = (1-d) + d * ((PR(B)/1 + PR(C)/1 + PR(D)/1) / 3)
    2: 
        B i D mają ten sam PageRank, ponieważ tylko węzeł C je linkuje (ma do nich połączenia)
        PR(B) = (1-d) + d * ((PR(C)/1) / 1)
        PR(D) = (1-d) + d * ((PR(C)/1) / 1)
        PR(B) = PR(D)
    4: 
        Macierz a:
            [[0, 1/3, 1/4, 1/3, 0],
            [0, 0, 1/4, 0, 0],
            [0, 1/3, 0, 1/3, 0],
            [0, 0, 1/4, 0, 0],
            [0, 1/3, 1/4, 1/3, 0]]
        PageRank dla każdego węzła: [0.00533815 0.00254648 0.00357289 0.00254648 0.00533815]
"""
