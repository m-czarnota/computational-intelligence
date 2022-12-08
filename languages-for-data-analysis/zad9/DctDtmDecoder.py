import numpy as np
import scipy


class DctDtmDecoder:
    def decode(self, filename: str):
        blocks = self.read_from_file(filename)
        print(blocks)

    @staticmethod
    def read_from_file(filename: str):
        blocks = []
        block_size = 0
        is_block = False
        actual_block = []
        content = ''

        file = open(filename)
        for line in file:
            line = line.replace('\n', '')
            line = line.strip()
            line = line.replace('  ', ' ')
            # print(line, line.split(' '))

            if 'nan' in line:
                block_size = int(line.replace('nan', ''))
                blocks.append(np.full((block_size, block_size), np.nan))
                continue

            if '[' in line:
                is_block = True
                line = line.replace('[', '')

                block = list(map(lambda x: float(x), line.split(' ')))
                block.extend([0] * (block_size - len(block)))
                actual_block.extend(list(map(lambda x: float(x), line.split(' '))))

                continue

            if ']' in line:
                is_block = False
                line = line.replace(']', '')

                actual_block.extend(list(map(lambda x: float(x), line.split(' '))))
                blocks.append(np.array(actual_block).resize((block_size, block_size)))
                actual_block = []

                continue

            if is_block:
                actual_block.extend(list(map(lambda x: float(x), line.split(' '))))
                continue

        file.close()

        return np.array(blocks)

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
