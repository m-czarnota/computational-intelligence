import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
import time

DATA_FOLDER = './data'
IMAGES_FOLDER = './images'


class Seabed:
    def __init__(self, grid_resolution: float = 1.0, circle_size: float = 1.0, min_points_count: int = 2, power: float = 2.0):
        self.grid_resolution = grid_resolution
        self.circle_size = circle_size
        self.min_points_count = min_points_count
        self.power = power

        self.passed_filename = None
        self.default_filename = 'wraki utm.txt'
        self.grid = None
        self.x_min_max = {'min': 0, 'max': 0}
        self.y_min_max = {'min': 0, 'max': 0}
        self.grid_shape = [0, 0]

    def calc_grid(self, filepath: str = None):
        data = np.loadtxt(filepath) if filepath else self.read_default_data()
        self.passed_filename = str(filepath if filepath else self.default_filename).split('/')[-1].split('.')[0]

        self.calc_mins_maxes(data)
        self.grid = np.zeros(self.grid_shape)
        kd_tree = KDTree(data[:, :2])

        for x_iter, x_point in enumerate(np.linspace(self.x_min_max['min'], self.x_min_max['max'], self.grid_shape[0])):
            print(f'Progress: {x_iter + 1}/{self.grid_shape[0]}')

            for y_iter, y_point in enumerate(np.linspace(self.y_min_max['min'], self.y_min_max['max'], self.grid_shape[1])):
                indexes, distances = kd_tree.query_radius([[x_point, y_point]], r=self.circle_size, return_distance=True)
                indexes = indexes[0]
                distances = distances[0]

                if len(indexes) < self.min_points_count:
                    self.grid[x_iter, y_iter] = np.NaN
                    continue

                powered_distances = distances ** self.power
                measured_values = data[indexes][:, 2]

                nominator = np.sum(np.divide(measured_values, powered_distances, out=np.zeros_like(measured_values), where=powered_distances != 0))
                denominator = np.sum(np.divide(1, powered_distances, out=np.zeros_like(distances), where=powered_distances != 0))
                estimated_val = nominator / denominator

                self.grid[x_iter, y_iter] = estimated_val

        return self.grid

    def calc_mins_maxes(self, data: np.array):
        self.x_min_max['min'] = np.min(data[:, 0])
        self.x_min_max['max'] = np.max(data[:, 0])
        self.grid_shape[0] = int(np.ceil((self.x_min_max['max'] - self.x_min_max['min']) / self.grid_resolution))

        self.y_min_max['min'] = np.min(data[:, 1])
        self.y_min_max['max'] = np.max(data[:, 1])
        self.grid_shape[1] = int(np.ceil((self.y_min_max['max'] - self.y_min_max['min']) / self.grid_resolution))

    def save_grid_to_file(self):
        with open(f'{DATA_FOLDER}/{self.generate_name_to_save()}.asc', 'w') as f:
            f.write(f'ncols {self.grid.shape[1]}\n')
            f.write(f'nrows {self.grid.shape[0]}\n')
            f.write(f'xllcenter {self.x_min_max["min"]}\n')
            f.write(f'yllcenter {self.y_min_max["min"]}\n')
            f.write(f'cellsize {self.grid_resolution}\n')
            f.write(f'nodata_value {np.NaN}\n')

            for row in self.grid:
                f.write(' '.join(np.char.mod('%f', row)) + '\n')

    def visualise_grid_2d(self, save_to_file: bool = True):
        x = np.arange(self.x_min_max['min'], self.x_min_max['max'], self.grid_resolution)
        y = np.arange(self.y_min_max['min'], self.y_min_max['max'], self.grid_resolution)
        [xx, yy] = np.meshgrid(x, y)

        plt.figure(figsize=(20, 10))
        plt.contourf(xx, yy, self.grid.T)

        if save_to_file:
            filename = self.generate_name_to_save()
            plt.savefig(f'{IMAGES_FOLDER}/{filename}.png')
            return

        plt.show()

    def read_default_data(self):
        return np.loadtxt(f'{DATA_FOLDER}/{self.default_filename}')

    def generate_name_to_save(self):
        return f'{self.passed_filename}_grid_{self.grid_resolution}_circle_{self.circle_size}_points_{self.min_points_count}_power_{self.power}'


if __name__ == '__main__':
    filepath = f'{DATA_FOLDER}/UTM-obrotnica.txt'
    seabed = Seabed(grid_resolution=1.0)

    t1 = time.time()
    seabed.calc_grid(filepath)
    t2 = time.time()
    print(f'Time: {t2 - t1}s')

    seabed.save_grid_to_file()
    seabed.visualise_grid_2d()

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
