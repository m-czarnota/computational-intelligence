import numpy as np
import scipy


class DctDtmDecoder:
    def decode(self, filename: str):
        vectors = self.read_from_file(filename)
        blocks = []

        for vector in vectors:
            block = self.map_vector_by_zigzag_to_block(vector)
            idct_block = self.idct(block)

            blocks.append(idct_block)

        return np.array(blocks)

    def read_from_file(self, filename: str):
        vectors = []
        actual_vector = []

        is_vector = False
        block_size = 0

        file = open(filename)
        for line in file:
            line = line.replace('\n', '')
            line = line.strip()
            line = line.replace('  ', ' ')

            if 'nan' in line:
                block_size = int(line.replace('nan', ''))
                vectors.append(np.full(block_size ** 2, np.nan))
                continue

            if '[' in line:
                is_vector = True
                line = line.replace('[', '')

                numbers = self.get_numbers_from_line(line)
                actual_vector.extend(numbers)

                continue

            if ']' in line:
                is_vector = False
                line = line.replace(']', '')

                numbers = self.get_numbers_from_line(line)
                actual_vector.extend(numbers)

                extend_with_0_count = block_size ** 2 - len(actual_vector)
                actual_vector.extend([0] * extend_with_0_count)

                vectors.append(np.array(actual_vector))
                actual_vector = []

                continue

            if is_vector:
                numbers = self.get_numbers_from_line(line)
                actual_vector.extend(numbers)

                continue

        file.close()

        return np.array(vectors)

    @staticmethod
    def join_blocks_to_matrix(blocks: np.array, block_size: int):
        partial_width = int(8 / block_size)
        partial_height = int(8 / block_size)
        which_block = 0

    @staticmethod
    def get_numbers_from_line(line: str):
        return list(map(lambda x: float(x), line.split(' ')))

    @staticmethod
    def idct(array: np.array):
        return scipy.fftpack.idct(scipy.fftpack.idct(array.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

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
