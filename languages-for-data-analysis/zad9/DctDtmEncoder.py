import os
import numpy as np
import zipfile

import functions

DATA_FOLDER = './data'
IMAGES_FOLDER = './images'


class DctDtmEncoder:
    def __init__(self, filename_to_save: str = 'compressed_data'):
        self.filename_to_save: str = filename_to_save

    def encode(self, data: np.array, block_size: int = 32, compress_error: float = 0.05, zipping: bool = True):
        blocks = self.split_matrix_to_blocks(data, block_size)
        encoded_data = []

        for block_iter, block in enumerate(blocks):
            print(f'Encode progress: {((block_iter + 1) / blocks.shape[0] * 100):.2f}%')

            if np.isnan(block).any():
                encoded_data.append(f'{block_size}{np.NaN}')
                continue

            dct_block = functions.dct(block)
            vector = functions.map_block_by_zigzag_to_vector(dct_block)

            cropped_vector = self.crop_vector(vector, block, compress_error)
            encoded_data.append(cropped_vector)

        encoded_data = np.array(encoded_data)
        self.save_to_file(encoded_data, data.shape)

        if zipping:
            with zipfile.ZipFile(f'{DATA_FOLDER}/{self.filename_to_save}.zip', 'w', compression=zipfile.ZIP_BZIP2, compresslevel=9) as zf:
                zf.write(f'{DATA_FOLDER}/{self.filename_to_save}_zip.txt')
            os.remove(f'{DATA_FOLDER}/{self.filename_to_save}_zip.txt')

        return encoded_data

    def save_to_file(self, encoded_data: np.array, data_shape: tuple, zipping: bool = True):
        zip_suffix = '_zip' if zipping else ''

        with open(f'{DATA_FOLDER}/{self.filename_to_save}{zip_suffix}.txt', 'w') as f:
            f.write(f'{data_shape}\n'.replace('(', '').replace(')', ''))

            for data in encoded_data:
                if type(data) == str:
                    f.write(f'{data}\n')
                    continue

                f.write(' '.join(np.char.mod('%f', data)) + '\n')

    @staticmethod
    def split_matrix_to_blocks(data: np.array, block_size: int):
        blocks = []

        for which_row in range(0, data.shape[0], block_size):
            for which_column in range(0, data.shape[1], block_size):
                block = data[which_row: which_row + block_size, which_column: which_column + block_size]
                row_shape, column_shape = block.shape

                if column_shape < block_size:
                    for _ in range(block_size - column_shape):
                        block = np.c_[block, np.array([np.nan] * row_shape)]

                if row_shape < block_size:
                    for _ in range(block_size - row_shape):
                        block = np.vstack((block, np.array([np.nan] * block_size)))

                blocks.append(block)

        return np.array(blocks)

    @staticmethod
    def crop_vector(vector: np.array, original_block: np.array, compress_error: float):
        vector = np.copy(vector)
        index_modifier = 0.75

        index_to_cut = int(vector.shape[0] * index_modifier)
        iteration_stop = 100
        iter_count = 0

        while iter_count < iteration_stop:
            if index_to_cut == 0:
                index_to_cut += 1
                break

            iter_count += 1

            vector_copy = vector.copy()
            vector_copy[index_to_cut:] = 0

            reconstructed_block = functions.map_vector_by_zigzag_to_block(vector_copy)
            idct_block = functions.idct(reconstructed_block)
            subtraction_block = idct_block - original_block

            error = np.max(abs(subtraction_block))

            if error > compress_error:
                index_to_cut = int(index_to_cut / index_modifier)
                continue

            if error < compress_error:
                index_to_cut = int(index_to_cut / (index_modifier * 2))
                continue

            break

        return vector[:index_to_cut]
