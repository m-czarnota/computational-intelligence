import os
import time
import zipfile
import numpy as np
import scipy

DATA_FOLDER = './data'
IMAGES_FOLDER = './images'


class DctDtmEncoder:
    def __init__(self, filename_to_save: str = 'compressed_data'):
        self.filename_to_save = filename_to_save

    def encode(self, data: np.array, block_size: int = 16, compress_accuracy: float = 5.0, zipping: bool = True):
        blocks = self.split_matrix_to_blocks(data, block_size)
        encoded_data = []
        start_calc_time = time.time()

        for block_iter, block in enumerate(blocks):
            print(f'Encode progress: {((block_iter + 1) / blocks.shape[0] * 100):.2f}%')

            if np.isnan(block).any():
                encoded_data.append(f'{block_size}{np.NaN}')
                continue

            dct_block = self.dct(block)
            vector = self.map_block_by_zigzag_to_vector(dct_block)
            encoded_data.append(vector[:np.ceil(vector.shape[0] / compress_accuracy).astype(np.int16)])

        end_calc_time = time.time()
        encoded_data = np.array(encoded_data)
        self.save_to_file(encoded_data)

        if zipping:
            with zipfile.ZipFile(f'{DATA_FOLDER}/{self.filename_to_save}.zip', 'w') as zf:
                zf.write(f'{DATA_FOLDER}/{self.filename_to_save}.txt')
            os.remove(f'{DATA_FOLDER}/{self.filename_to_save}.txt')

        return end_calc_time - start_calc_time

    def save_to_file(self, encoded_data: np.array):
        with open(f'{DATA_FOLDER}/{self.filename_to_save}.txt', 'w') as f:
            for data in encoded_data:
                f.write(f'{data}\n')

    @staticmethod
    def split_matrix_to_blocks(data: np.array, block_size: int):
        blocks = []

        for which_row in range(0, data.shape[0], block_size):
            for which_column in range(0, data.shape[1], block_size):
                blocks.append(data[which_row: which_row + block_size, which_column: which_column + block_size])

        return np.array(blocks)

    @staticmethod
    def dct(array: np.array):
        return scipy.fftpack.dct(scipy.fftpack.dct(array.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

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
