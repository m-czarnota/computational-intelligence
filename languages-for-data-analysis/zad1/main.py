import numpy as np


def zad1(matrix, n):
    shape = matrix.shape
    matrix_copy = matrix.copy()

    for i in range(0, n):
        row = np.zeros((1, shape[1]))
        matrix = np.vstack((row, matrix))
        matrix = np.vstack((matrix, row))

    shape = matrix.shape
    for i in range(0, n):
        column = np.zeros((1, shape[0])).transpose()
        matrix = np.hstack((column, matrix))
        matrix = np.hstack((matrix, column))

    return matrix


def zad2(n: int):
    pass


def zad3(matrix, a):
    matrix_copy = matrix.copy()
    matrix_copy[(matrix_copy >= -np.abs(a)) & (matrix_copy <= np.abs(a))] = 0
    return matrix_copy


def zad4(matrix, a):
    matrix_copy = matrix.copy()
    for index, row in enumerate(matrix_copy):
        if (-np.abs(a) in row) or (np.abs(a) in row):
            matrix_copy = np.delete(matrix_copy, index, 0)
    return matrix_copy


def zad5(*arg):
    uniques = set()
    for matrix in arg:
        [uniques.add(value) for value in np.unique(matrix)]
    return sorted(uniques)


if __name__ == '__main__':
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print('zad1\n', zad1(matrix, 2))
    print('zad2\n', zad2(5))
    print('zad3\n', zad3(matrix, 2))
    print('zad4\n', zad4(matrix, 2))
    print('zad5\n', zad5(matrix, [[1, 4, 64, 234], [654, 234, 2, 3]]))
