"""
dla każdego k powtórzyć eksperyment kilka razy i usrednić, żeby wykres nie był poszarpany
na wykresie zaznaczamy optymalną wartosć parametru k
powinno być conajmniej 1000 pkt

dzielimy zbiór na pół. y_test to wyniki dla idealnej
sprawdzać od k=1 do k=rozmiar_proby, jak rozmiar_proby == 500, to 500
dla małych k dać krok mały, dla większych dać większy - do 20 co 2, do 50 co 5 na przykład
"""


import math
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns

SAMPLE_COUNT = 1000
IMAGES_FILEPATH = './images'


def generate_dim(n: int, gauss: bool = False):
    x_val = 0.5
    coefficients = [np.random.rand() for i in range(n - 1)]
    a0 = x_val - np.sum([coefficient * x_val for coefficient in coefficients])

    x = np.random.rand(SAMPLE_COUNT, n)
    y = np.array([coefficient * x[:, 0] + a0 for coefficient in coefficients][0])
    y[y > x[:, 1]] = 1
    y[y <= x[:, 1]] = 0

    if gauss is True:
        std = get_truncated_normal(low=0, upp=0.5)
        for i in range(n):
            x[:, i] += std.rvs()

    return x, y


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_chessboard(n: int, gauss: bool = False):
    second_dim = math.sqrt(n)

    x = np.random.rand(SAMPLE_COUNT, 2)

    y = np.add(((np.floor(x[:, 0] * second_dim)) % second_dim), ((np.floor(x[:, 1] * second_dim)) % second_dim))
    y = y % 2

    if gauss is True:
        std = get_truncated_normal(low=0, upp=0.5)
        x[:, 0] += std.rvs()
        x[:, 1] += std.rvs()

    return x, y


def zad2(x: np.ndarray, y: np.ndarray, dim: int, dataset: str, gauss: bool = False):
    k_dict = {}
    k = 1
    k_step = 10
    max_score = 0
    k_optimal = 0

    for i in range(1, SAMPLE_COUNT):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)

        new_score = neigh.score(X_test, y_test)
        if k in k_dict.keys():
            k_dict[k].append(new_score)
        else:
            k_dict[k] = [new_score]

        if new_score > max_score:
            max_score = new_score
            k_optimal = k

        if i % k_step == 0:
            k += 1

    print(f'{dataset}, samples={SAMPLE_COUNT}, dim={dim}, gauss={gauss}, max_score={max_score}, the best k={k_optimal}')
    for key, values in k_dict.items():
        k_dict[key] = np.mean(values)

    # plt.figure()
    # plt.scatter(k_dict.keys(), k_dict.values(), label='mean score in k')
    # plt.scatter(k_optimal, max_score, c='r', label='max score')
    # plt.legend()
    # plt.title(f'score by k; k is avg for different data splits, k_step={k_step}; the best k={k_optimal}, max_score={max_score}')
    # plt.xlabel('k')
    # plt.ylabel('score')
    # plt.grid()
    # plt.savefig(f'{IMAGES_FILEPATH}/zad2_{dataset}_samples_{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}.png')
    # plt.show()


def zad3(x: np.ndarray, y: np.ndarray, dim: int, dataset: str, gauss: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    k = 6
    algorithms = ['brute', 'kd_tree', 'ball_tree']

    times_of_fit = {}
    times_of_score = {}
    scores = []

    for algorithm in algorithms:
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm)

        time1 = time.time()
        neigh.fit(X_train, y_train)
        time2 = time.time()
        times_of_fit[algorithm] = time2 - time1

        time1 = time.time()
        score = neigh.score(X_test, y_test)
        time2 = time.time()
        times_of_score[algorithm] = time2 - time1
        scores.append(score)

    plt.figure()
    plt.title(f'differences in time of fit by algorithm, k={k}')
    plt.bar(times_of_fit.keys(), times_of_score.values(), label='time of fit by algorithm')
    plt.savefig(f'{IMAGES_FILEPATH}/zad3_{dataset}_samples_{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}-algorithms_times_k_{k}.png')
    # plt.show()

    plt.figure()
    plt.title(f'differences in time of score by algorithm, k={k}')
    plt.bar(times_of_score.keys(), times_of_score.values())
    plt.savefig(f'{IMAGES_FILEPATH}/zad3_{dataset}_samples_{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}-algorithms_score_time_k_{k}.png')
    # plt.show()

    plt.figure()
    plt.title(f'differences in scores by algorithm, k={k}')
    plt.bar(times_of_score.keys(), scores, label='score by algorithm')
    plt.savefig(f'{IMAGES_FILEPATH}/zad3_{dataset}_samples_{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}-algorithms_scores_k_{k}.png')
    # plt.show()


def zad4(x: np.ndarray, y: np.ndarray, dim: int, dataset: str, gauss: bool = False):
    algorithms = ['kd_tree', 'ball_tree']
    k = 6
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    for algorithm in algorithms:
        times_of_fit = {}
        times_of_score = {}
        scores = []

        for leaf_size in range(1, 60):
            neigh = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, leaf_size=leaf_size)

            time1 = time.time()
            neigh.fit(x, y)
            time2 = time.time()
            times_of_fit[leaf_size] = time2 - time1

            time1 = time.time()
            score = neigh.score(X_test, y_test)
            time2 = time.time()
            times_of_score[leaf_size] = time2 - time1
            scores.append(score)

        plt.figure()
        plt.title(f'differences by leaf size for algorithm={algorithm}, k={k}')
        plt.xlabel('leaf size')
        plt.ylabel('time')
        plt.plot(times_of_fit.keys(), times_of_fit.values(), label='time of fit by leaf size')
        plt.plot(times_of_score.keys(), times_of_score.values(), label='time of score by leaf size')
        plt.legend()
        plt.savefig(f'{IMAGES_FILEPATH}/zad4_{dataset}_samples_{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}-leaf_size_{algorithm}_k_{k}.png')
        # plt.show()

        plt.figure()
        plt.title(f'differences in scores by leaf size for algorithm={algorithm}, k={k}')
        plt.xlabel('leaf size')
        plt.ylabel('score')
        plt.scatter(times_of_score.keys(), scores, label='score by leaf size')
        plt.legend()
        plt.savefig(f'{IMAGES_FILEPATH}/zad4_{dataset}_samples_{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}-leaf_size_scores_{algorithm}_k_{k}.png')
        # plt.show()


def zad5(x: np.ndarray, y: np.ndarray, dim: int, dataset: str, gauss: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    k = (1, 3, 11, 25)

    for val_k in k:
        neigh = KNeighborsClassifier(n_neighbors=val_k)
        neigh.fit(X_train, y_train)

        DecisionBoundaryDisplay.from_estimator(neigh, X_train, response_method='predict', plot_method="pcolormesh")
        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=neigh.predict(X_test), alpha=1.0, edgecolor='black')
        plt.savefig(f'{IMAGES_FILEPATH}/zad5_{dataset}_samples{SAMPLE_COUNT}_dim_{dim}_gauss_{gauss}-separation_boundary_k_{k}.png')
        # plt.show()


if __name__ == '__main__':
    dice_dims = (2, 3, 6, 10)
    chessboard_dims = (4, 9, 16, 25)

    # for dice_dim, chessboard_dim in zip(dice_dims, chessboard_dims):
    #     [X, Y] = generate_dim(dice_dim)
    #     zad2(X, Y, dice_dim, 'dice')
    #
    #     [X, Y] = generate_chessboard(chessboard_dim)
    #     zad2(X, Y, chessboard_dim, 'chessboard')

    # for i, chessboard_dim in enumerate(chessboard_dims):
    #     if i == len(dice_dims) - 1:
    #         break
    #
    #     [X, Y] = generate_dim(chessboard_dim)
    #     zad3(X, Y, chessboard_dim, 'chessboard')

    # [X, Y] = generate_chessboard(chessboard_dims[3])
    # zad4(X, Y, chessboard_dims[3], 'chessboard')

    [X, Y] = generate_dim(dice_dims[0])
    zad5(X, Y, dice_dims[0], 'dice')

    for dim in chessboard_dims:
        [X, Y] = generate_chessboard(dim)
        zad5(X, Y, dim, 'chessboard')


"""
results from zad2:
    dice, samples=1000, dim=2, gauss=False, max_score=0.998, the best k=28
    chessboard, samples=1000, dim=4, gauss=False, max_score=0.992, the best k=10
    dice, samples=1000, dim=3, gauss=False, max_score=0.992, the best k=5
    chessboard, samples=1000, dim=9, gauss=False, max_score=0.962, the best k=1
    dice, samples=1000, dim=6, gauss=False, max_score=1.0, the best k=1
    chessboard, samples=1000, dim=16, gauss=False, max_score=0.934, the best k=1
    dice, samples=1000, dim=10, gauss=False, max_score=1.0, the best k=1
    chessboard, samples=1000, dim=25, gauss=False, max_score=0.904, the best k=1

----------------------------------------------------------------------------------

results from zad3:
    chessboard_samples_1000_dim_4_gauss_False k=6:
        czas wyznaczania dokładności klasyfikacji był znacznie dłuższy dla algorytmu indeksowania 'brute'
        w pozostałych przypadkach dla 'kd_tree', 'ball_tree' czas ten był znacząco krótszy, co oznacza, że wykorzystanie sprytnego algorytmu przyspiesza pracę niż siłowe próby
        
        w przypadku dokładności klasyfikacji wyniki nie różniły się od siebie znacząco i były do siebie bardzo podobne dla każdego z algorytmów indeksowania
        
        czas nauczania był znacznie dłuższy dla 'brute' od pozostałych przypadków.
        'kd_tree' oraz 'ball_tree' miały prawie identyczne czasy w nauczaniu
        
    chessboard_samples_1000_dim_9_gauss_False k=6:
        czas wyznaczania dokładności klasyfikacji był krótszy dla 'brute' niż w poprzednim przypadku, jednak wciąż znacznie dłuższy niż przy pozostałych algorytmach.
        pozostałe algorytmy miały podobne czasy, jednak 'ball_tree' był szybszy
        
        dokładność klasyfikacji - wyniki były takie same dla wszystkich algorytmów indeksowania
        
        czasy nauczania są podobne jak czasy wyznaczania dokładności klasyfikacji
        
    chessboard_samples_1000_dim_16_gauss_False k=6:
        czas wyznaczania dokładności klasyfikacji jeszcze krótszy niż w poprzednim przypadku z mniejszą wymiarowością, jednak wciąż znacząco odstający od pozostałych
        dla pozostałych algorytmów indeksowania czasy są podobne do tych w poprzednim przypadku
        
        dokładność klasyfikacji - wyniki były takie same dla wszystkich algorytmów indeksowania
        
        czasy nayczania są podobne jak czasy wyznaczania dokładności klasyfikacji

----------------------------------------------------------------------------------

results from zad4:
    chessboard_samples_5000_dim_25_gauss_False k=6:
        czas nauczania dla 'ball_tree' jest podobny do czasu nauczania dla 'kd_tree'. dla 'kd_tree' przy małej liczbie liści jest on wyższy niż w przypadku małej ilości liści dla 'ball_tree'
        czas nauczania jest podobny na całym zakresie ilości badanych liści od 1 do 60
        aż do rozmiaru liści 17 czas nauczania dla 'ball_tree' nie zmienia się. w przypadku 'kd_tree' jest on bardziej stały dla ilości liści od 10 do 29
        
        czas wyznaczania dokładności klasyfikacji jest podobny dla obydwu algorytmów indeksowania niezależnie od ilości liści
        dla 'ball_tree' wraz ze wzrostem liczby liści jest widoczny zwiększony czas wyznaczania dokładności klasyfikacji
        przy 'kd_tree' w okolicy ilości liści równej 8 czas wyznaczania dokładności klasyfikacji wzrósł, co jest widoczne jako pik na wykresie. przy wyższej ilości liści czas wyznaczania dokładności klasyfikacji jest na podobnym poziomie i nie rośnie jak w przypadku 'ball_tree'

    chessboard_samples_5000_dim_25_gauss_False k=6:
        brak różnic w dokładności klasyfikacji niezależnie od algorytmu oraz ilości liści
        
----------------------------------------------------------------------------------

summary conclusions:
    a) złożoność obliczeniowa:
        oznaczenia:
            * n - liczba próbek
            * d - wymiar
            * k - liczba sąsiadów
        - brute:
            * czas nauczania: O(1)
            * czas wyznaczania dokładności klasyfikacji: O(k * n * d)
        - kd_tree:
            * czas nauczania: O(d * n * log(n))
            * czas wyznaczania dokładności klasyfikacji: O(k * log(n))
        - ball_tree:
            * czas nauczania: O(d * n * log(n))
            * czas wyznaczania dokładności klasyfikacji: O(k * log(n))
            
        z zebranych wyników wynika, że ball_tree jest delikatnie szybszy od kd_tree
        
    b) czas wykonania:
        na podstawie zgromadzonych wyników z badań wynika, że algorytm indeksowania brute jest znacznie wolniejszy od pozostałych algorytmów indeksowania
        ball_tree jest delikatnie szybszy od kd_tree
        
    c) w przypadku szachownicy im mniej kwadratów szachownicy, tym większe k dla osiągnięcia lepszej dokładności klasyfikacji.
    im więcej kwadratów w szachownicy tym dla mniejszego k jest osiągana lepsza dokładność
        
"""
