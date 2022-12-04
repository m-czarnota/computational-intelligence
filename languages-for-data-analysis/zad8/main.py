import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
import time

DATA_FOLDER = './data'


def grid():
    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    rows_count = int(np.ceil((x_max - x_min) * grid_resolution))
    print(x_min, x_max, rows_count)

    y_min = np.min(data[:, 1])
    y_max = np.max(data[:, 1])
    columns_count = int(np.ceil((y_max - y_min) * grid_resolution))
    print(y_min, y_max, columns_count)
    print(data.shape)

    array = np.zeros((rows_count, columns_count))

    for x_iter in np.arange(rows_count):
        x_index = x_min + x_iter

        for y_iter in np.arange(columns_count):
            y_index = y_min + y_iter

            indexes, distances = kd_tree.query_radius([[x_index, y_index]], r=circle_size, return_distance=True)
            indexes = indexes[0]
            distances = distances[0]
            # print(data[indexes][:, 2], distances)

            estimated_val = np.NaN

            if len(indexes) >= min_points_count:
                powered_distances = distances ** power
                measured_values = data[indexes][:, 2]

                nominator = np.sum(np.divide(measured_values, powered_distances, out=np.zeros_like(measured_values), where=powered_distances != 0))
                denominator = np.sum(np.divide(1, powered_distances, out=np.zeros_like(distances), where=powered_distances != 0))
                estimated_val = nominator / denominator

            array[x_iter, y_iter] = estimated_val

    print(array[array != np.NaN].shape)
    return array


def save_grid_to_file(array: np.array):
    with open('my_grid.asc', 'w') as f:
        f.write(f'ncols {array.shape[1]}\n')
        f.write(f'nrows {array.shape[0]}\n')

        f.write(f'cellsize {grid_resolution}\n')
        f.write(f'nodata_value {np.NaN}')


if __name__ == '__main__':
    data = np.loadtxt(f'{DATA_FOLDER}/wraki utm.txt')
    # data = data[:, :2]
    # print(data)
    kd_tree = KDTree(data[:, :2])

    grid_resolution = 1
    circle_size = 15
    min_points_count = 3
    power = 2

    t1 = time.time()
    array = grid()
    t2 = time.time()
    print(f'Time: {t2 - t1}s')

    # print(array)
    plt.figure()
    plt.scatter(array[:, 0], array[:, 1])
    plt.show()

"""
utm-brama, utm-obrotnica, wraki utm 
wraki utm jest plikiem dużo mniejszym

przykladowe-dane-GRID pokazują jak zapisać dane
number cols, number rows to rozmiary macierzy
xllcenter, yllcenter to wartość pkt startu w danych metrycznych - zawsze my mamy to samo
cellsize to my ustawiamy
nodata value to u nas NaN

to co jest na czerwono jest wyżej punktowane, bez czerwonych max ocena 4
im więcej czerwonych tym lepiej. jak idziemy w czerwone to odpuszczamy niebieskie
user ustawia rozdzielczość grid, czy to 1m, 2m, 5m. to jako jedna zmienna
określenie rozmiaru okienka - albo po kwadratach albo po okienkach
minimalna liczba pkt do obliczeń
nie korzystamy z bibliotek i gotowych rozwiązań dla niebieskich, dozwolone tylko dla czerwonych. nie może być jedna funkcja, tylko samemu

jak mamy milion pomiarów można zrobić pętle, ale będzie się długo liczyć
aby było szybciej można wykorzystać kd tree
w kd tree elementy na dole są tak ułożone że na dole będą wartości leżące w pobliżu siebie
napisać to jedną funkcją

wczytać dane
wyznaczyć wartości graniczne x_min, x_max
na podstawie parametrów określić rozmiar macierzy grid
dane wczytać do kd_tree
w pętli po x, y pobierać z kd_tree wartości, wyliczać i zapisywać w konkretnym miejscu
powinno w miarę szybko się liczyć. powinno się to liczyć kilka sekund, kilkanaście sekund
z kd_tree okręgi będą prostsze niż kwadraty i są bardziej preferowane
stalamy w jakim otoczeniu szukamy punktu
kd_tree ma takie metody jak znajdź wszytkie najbliższe pkt leżące w danej odległości, znajdź najbliższe punkty dla danego punktu

jak mam prostokąt 
    480
100     250
    300
i mam rozmiar okna 0.5
to liczę grida o rozmiarze 300x360 (250 - 100)*0.5, (480-300)*0.5

dla moving average: jak kd_tree zwróci mi 5 wartości, to liczę z nich średnią
pierwszy pkt w grid odpowiada x_min, następny odpowiada (x_min + x) / x_max

na początku bawić się na wrakach
dostajemy pliki metryczne UTM (M to jest metric)
co to znaczy? 
    dx = |x2 - x1|
    dy = |y2 - y1|
    d = sqrt(dx**2 + dy**2)
"""
