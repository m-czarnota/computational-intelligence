import numpy as np
import cv2
import zipfile
import time
import os
import scipy

DATA_FOLDER = './data'
IMAGES_FOLDER = './images'


class DctDtm:
    def __init__(self):
        pass

    def compress(self, data: np.array, block_size: int = 16, compress_accuracy: float = 5, zipping: bool = True):
        blocks = self.split_matrix_to_blocks(data, block_size)

        compressed_data = []
        for block in blocks:
            if np.isnan(block).any():
                compressed_data.append(f'{block_size}{np.NaN}')
                continue

            dct_block = self.dct(block)
            vector = self.map_block_by_zigzag_to_vector(dct_block)
            compressed_data.append(vector[:np.ceil(vector.shape[0] / compress_accuracy).astype(np.int16)])

    @staticmethod
    def split_matrix_to_blocks(data: np.array, block_size: int):
        blocks = []

        for which_row in range(0, data.shape[0], block_size):
            for which_column in range(0, data.shape[1], block_size):
                blocks.append(data[which_row: which_row + block_size, which_column: which_column + block_size])

        return np.array(blocks)

    @staticmethod
    def dct(array: np.array):
        return scipy.fftpack.dct(
            scipy.fftpack.dct(
                array.astype(float),
                axis=0,
                norm='ortho'
            ),
            axis=1,
            norm='ortho'
        )

    @staticmethod
    def map_block_by_zigzag_to_vector(block: np.array):
        vector = []
        row = 0
        col = 0
        going_down = False

        while row < block.shape[0] and col < block.shape[1]:
            vector.append(block[row, col])

            if going_down:
                if row == block.shape[0] - 1 or col == 0:
                    going_down = False

                    if row == block.shape[0] - 1:
                        col += 1
                    else:
                        row += 1
                else:
                    row += 1
                    col -= 1
            else:
                if row == 0 or col == block.shape[1] - 1:
                    going_down = True

                    if col == block.shape[1] - 1:
                        row += 1
                    else:
                        col += 1
                else:
                    row -= 1
                    col += 1

        return np.array(vector)

    @staticmethod
    def map_vector_by_zigzag_to_block(vector: np.array):
        block_size = np.sqrt(vector.shape[0]).astype(np.int16)
        block = np.zeros((block_size, block_size))

        row = 0
        col = 0
        going_down = False

        for val in vector:
            block[row, col] = val

            if going_down:
                if row == block_size - 1 or col == 0:
                    going_down = False

                    if row == block_size - 1:
                        col += 1
                    else:
                        row += 1
                else:
                    row += 1
                    col -= 1
            else:
                if row == 0 or col == block_size - 1:
                    going_down = True

                    if col == block_size - 1:
                        row += 1
                    else:
                        col += 1
                else:
                    row -= 1
                    col += 1

        return np.array(block)


if __name__ == '__main__':
    data = np.loadtxt(f'{DATA_FOLDER}/wraki utm_grid_0.1_circle_1.0_points_2_power_2.0.asc', skiprows=6)
    dct_dtm = DctDtm()
    # dct_dtm.compress(data)

    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
    print(DctDtm.map_vector_by_zigzag_to_block(DctDtm.map_block_by_zigzag_to_vector(a)))

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