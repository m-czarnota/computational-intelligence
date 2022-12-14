import numpy as np
import pandas as pd
import time
import os
from matplotlib import pyplot as plt

import functions
from DctDtmDecoder import DctDtmDecoder
from DctDtmEncoder import DctDtmEncoder

DATA_FOLDER = './data'
IMAGES_FOLDER = './images'


def zigzag_test():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

    print(functions.map_vector_by_zigzag_to_block(functions.map_block_by_zigzag_to_vector(a)))


def get_val_from_pd_series(series: pd.Series, str_to_cut: str):
    return series.map(lambda val: float(val.replace(f'{str_to_cut} ', ''))).values[0]


def encode():
    dct_dtm_encoder = DctDtmEncoder(filename)

    t1 = time.time()
    dct_dtm_encoder.encode(data, block_size=32, zipping=zipping)
    t2 = time.time()
    print(f'Encoding time: {t2 - t1}s')


def decode():
    dct_dtm_decoder = DctDtmDecoder()

    t1 = time.time()
    decoded = dct_dtm_decoder.decode(f'{DATA_FOLDER}/{filename}.{extension}', zipping)
    t2 = time.time()
    print(f'Decoding time: {t2 - t1}s')

    return decoded


def plot_seabed():
    x = np.arange(original_x_min, original_x_max, grid_resolution)
    y = np.arange(original_y_min, original_y_max, grid_resolution)
    [xx, yy] = np.meshgrid(x, y)

    plt.figure(figsize=(20, 10))
    plt.contourf(xx, yy, data.T)
    plt.title('Original seabed')
    # plt.show()
    plt.savefig(f'{IMAGES_FOLDER}/{filename}_original.png')

    plt.figure(figsize=(20, 10))
    plt.contourf(xx, yy, decoded_data.T)
    plt.title('Reconstructed seabed')
    # plt.show()
    plt.savefig(f'{IMAGES_FOLDER}/{filename}_reconstructed.png')


def plot_histogram():
    matrix_error = decoded_data - data
    hist, bins = np.histogram(matrix_error, range=(np.nanmin(matrix_error), np.nanmax(matrix_error)), density=True)

    plt.figure()
    plt.hist(hist, bins=bins)
    plt.title('Error distribution excluding NaN values')
    # plt.show()
    plt.savefig(f'{IMAGES_FOLDER}/{filename}_histogram.png')


if __name__ == '__main__':
    # zigzag_test()

    filename = 'wraki utm_grid_0.1_circle_1.0_points_2_power_2.0'
    filepath = f'{DATA_FOLDER}/{filename}.asc'
    data = np.loadtxt(filepath, skiprows=8)

    data_pd = pd.read_csv(filepath)
    original_x_min = get_val_from_pd_series(data_pd.iloc[1], 'xllmin')
    original_x_max = get_val_from_pd_series(data_pd.iloc[2], 'xllmax')
    original_y_min = get_val_from_pd_series(data_pd.iloc[3], 'yllmin')
    original_y_max = get_val_from_pd_series(data_pd.iloc[4], 'yllmax')
    grid_resolution = get_val_from_pd_series(data_pd.iloc[5], 'cellsize')

    zipping = True
    # encode()

    extension = "zip" if zipping else "txt"
    original_size = os.path.getsize(f'{DATA_FOLDER}/{filename}.asc')
    compressed_size = os.path.getsize(f'{DATA_FOLDER}/{filename}.{extension}')
    print(f'Compression ratio: {(original_size / compressed_size):.2f}:1')

    decoded_data = decode()
    plot_seabed()
    plot_histogram()

"""
na potrzeby zajęć wygenerować takie powierzchnie, które mają po 1000 w x i y
dla wraków może być 0.1, dla reszty 1m
radius minimum 1m, ilość pkt minimalnych 2 i w miarę powinno to być wypełnione
jak dla bramy damy 0.5 grid_resolution to będzie 2 razy większe

rozmiar bloku danych jako zmienna. testować 16x16 bloki, ale może być dowolny rozmiar
5cm, 10cm, 1cm, 0.1cm
z założenia zipujemy na koniec

dzielimy na bloki
sprawdzamy czy występuje w nim NaN. jak tak, to przepisujemy go albo pomijamy i w ogóle go nie kompresujemy.
jak są w całym bloku jakieś dane to je kompresujemy
po dekompresji obrazek bramy będzie schodkowy.

na koniec zamieniać macierz współczynników na wektor i na koniec bierzemy długość wektora - fajnie by było
można też po trójkącie i łatwiej będzie, a jeszcze łatwiej po kwadracie
pasek postępu

spakować zipem i zobaczyć czy to się poprawi
stopień kompresji w postaci 12:1, 15:1 - ile razy mniejszy jest plik
jeżeli są całe bloki NaN to my ich nie uwzględniamy przy liczeniu stopnia kompresji
mają być brane pod uwagę tylko bloki pełne
ale zapisujemy je, aby wyrównać do macierzy
na bieżąco można liczyć stopień kompresji przy blokach, wtedy będzie łatwiej

musimy zapisywać bloki NaN!
nie musimy zapisywać 16 NaN, można to skrócić i zapisać jako znacznik
jak jest pół na pół, albo dziura to traktujemy to jako blok z samymi NaN

stopień kompresji danych pełnych, czyli tylko tam gdzie są wszędzie dane
potem stopień kompresji całej macierzy
i na koniec stopień kompresji po przejechaniu zipem

musi powstać program dekompresujący
wczyta dane zdekompresowane i odtworzy powierzchnię
i wyświetli na ekranie
na ekranie mają się wyświetlać:
    powierzchnia po kompresji
    wykres błędów: po dekompresji - przed kompresją
nie wypisywać że kompresja jest 0.785, bo to nic nie mówi
można podawać stopień kompresji jako liczbę rzeczywistą 1.2:1

całe zadanie to odpowiednie skonstruowanie pętli i warunków
DCT oraz IDCT z bibliotek
metoda: idę po kolei aż osiągnę odpowiedni punkt

jak wektor utniemy, to przy odczytywaniu uzupełniamy go 0

"""
