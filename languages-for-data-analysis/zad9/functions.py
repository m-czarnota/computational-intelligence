import numpy as np
from scipy import fftpack


def dct(array: np.array):
    return fftpack.dct(fftpack.dct(array.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')


def idct(array: np.array):
    return fftpack.idct(fftpack.idct(array.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')


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
