import zipfile
import numpy as np
from io import TextIOWrapper

import functions


class DctDtmDecoder:
    def __init__(self):
        self.block_size: int = 0
        self.original_data_shape: tuple = ()

    def decode(self, filename: str, zipping: bool):
        vectors = self.read_encoded_data(filename, zipping)
        blocks = []

        for vector in vectors:
            block = functions.map_vector_by_zigzag_to_block(vector)
            idct_block = functions.idct(block)

            blocks.append(idct_block)

        data = self.join_blocks_to_matrix(blocks)

        return np.array(data)

    def read_encoded_data(self, filename: str, zipping: bool):
        if not zipping:
            with open(filename) as file:
                return self.read_from_file(file)

        with zipfile.ZipFile(filename) as z:
            inner_filename = filename.replace('zip', 'txt').replace('./', '')

            with TextIOWrapper(z.open(inner_filename)) as file:
                return self.read_from_file(file)

    def read_from_file(self, file):
        vectors = []
        self.block_size = 0

        for line_iter, line in enumerate(file):
            line_transformed = self.transform_line(line)

            if line_iter == 0:
                self.original_data_shape = tuple(map(lambda x: int(x.strip()), line_transformed.split(',')))
                continue

            if 'nan' in line_transformed:
                self.block_size = int(line_transformed.replace('nan', ''))
                vectors.append(np.full(self.block_size ** 2, np.nan))
                continue

            numbers = self.get_numbers_from_line(line)
            self.end_block(numbers, vectors)

        return np.array(vectors)

    def end_block(self, actual_vector: list, vectors: list):
        extend_with_0_count = self.block_size ** 2 - len(actual_vector)
        actual_vector.extend([0] * extend_with_0_count)

        vectors.append(np.array(actual_vector))

    def join_blocks_to_matrix(self, blocks: np.array):
        rows_to_add = self.block_size - (self.original_data_shape[0] % self.block_size)
        columns_to_add = self.block_size - (self.original_data_shape[1] % self.block_size)
        data = np.zeros((
            self.original_data_shape[0] + rows_to_add if rows_to_add != self.block_size else 0,
            self.original_data_shape[1] + columns_to_add if columns_to_add != self.block_size else 0,
        ))

        which_row = 0
        which_col = 0

        for block_iter, block in enumerate(blocks):
            row = which_row

            for block_row in block:
                col = which_col

                for val in block_row:
                    data[row, col] = val
                    col += 1

                row += 1

            which_col += self.block_size

            if which_col > self.original_data_shape[1]:
                which_row += self.block_size
                which_col = 0

        return data[:self.original_data_shape[0], :self.original_data_shape[1]]

    @staticmethod
    def transform_line(line: str):
        line_transformed = line.replace('\n', '').replace('\r', '')
        line_transformed = line_transformed.strip()

        return line_transformed

    @staticmethod
    def get_numbers_from_line(line: str):
        return list(map(lambda x: float(x), line.split(' ')))
