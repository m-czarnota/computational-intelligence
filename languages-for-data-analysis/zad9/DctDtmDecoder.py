import numpy as np

import functions


class DctDtmDecoder:
    def __init__(self):
        self.block_size: int = 0
        self.original_data_shape: tuple = ()

    def decode(self, filename: str):
        vectors = self.read_from_file(filename)
        blocks = []

        for vector in vectors:
            block = functions.map_vector_by_zigzag_to_block(vector)
            idct_block = functions.idct(block)

            blocks.append(idct_block)

        data = self.join_blocks_to_matrix(blocks)

        return np.array(data)

    def read_from_file(self, filename: str):
        vectors = []
        actual_vector = []

        is_vector = False
        self.block_size = 0

        file = open(filename)
        for line_iter, line in enumerate(file):
            line = line.replace('\n', '')
            line = line.strip()
            line = line.replace('  ', ' ')

            if line_iter == 0:
                self.original_data_shape = tuple(map(lambda x: int(x.strip()), line.split(',')))
                continue

            if 'nan' in line:
                self.block_size = int(line.replace('nan', ''))
                vectors.append(np.full(self.block_size ** 2, np.nan))
                continue

            if '[' in line:
                is_vector = True
                line = line.replace('[', '')

                if ']' in line:
                    line = line.replace(']', '')

                    numbers = self.get_numbers_from_line(line)
                    actual_vector.extend(numbers)

                    extend_with_0_count = self.block_size ** 2 - len(actual_vector)
                    actual_vector.extend([np.nan] * extend_with_0_count)

                    vectors.append(np.array(actual_vector))
                    actual_vector = []
                else:
                    numbers = self.get_numbers_from_line(line)
                    actual_vector.extend(numbers)

                continue

            if ']' in line:
                is_vector = False
                line = line.replace(']', '')

                numbers = self.get_numbers_from_line(line)
                actual_vector.extend(numbers)

                extend_with_0_count = self.block_size ** 2 - len(actual_vector)
                actual_vector.extend([np.nan] * extend_with_0_count)

                vectors.append(np.array(actual_vector))
                actual_vector = []

                continue

            if is_vector:
                numbers = self.get_numbers_from_line(line)
                actual_vector.extend(numbers)

                continue

        file.close()

        return np.array(vectors)

    def join_blocks_to_matrix(self, blocks: np.array):
        rows_to_add = self.block_size - (self.original_data_shape[0] % self.block_size)
        columns_to_add = self.block_size - (self.original_data_shape[1] % self.block_size)
        data = np.zeros((
            self.original_data_shape[0] + rows_to_add if rows_to_add != self.block_size else 0,
            self.original_data_shape[1] + columns_to_add if columns_to_add != self.block_size else 0,
        ))

        row = 0
        col = 0

        for block_iter, block in enumerate(blocks):
            if block_iter >= self.original_data_shape[0]:
                break

            which_row = np.copy(row)

            for block_row in block:
                col = 0 if col >= data.shape[1] else col

                for val in block_row:
                    data[which_row, col] = val
                    col += 1

                which_row += 1

            if block_iter % self.block_size == 0:
                row += 1

        return data[:self.original_data_shape[0], :self.original_data_shape[1]]

    @staticmethod
    def get_numbers_from_line(line: str):
        return list(map(lambda x: float(x), line.split(' ')))
