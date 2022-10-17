import cv2
import numpy as np
import time
from numba import jit, uint8, int32

DATA_FOLDER = './data/'
CLFS_FOLDER = './clasifiers/'

# each row describes white rectangle (in a template, in a unit square): (j, k, h, w)
HAAR_TEMPLATED = [
    np.array([[0.0, 0.0, 0.5, 1.0]]),  # "top-down edge" - punkt zaczepienia 0.0, wysokość 1/2, sięga do 1
    np.array([[0.0, 0.0, 1.0, 0.5]]),  # "left-right edge"
    np.array([[0.25, 0.0, 0.5, 1.0]]),  # "horizontal middle edge"
    np.array([[0.0, 0.25, 1.0, 0.5]]),  # "vertical middle edge"
    np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),  # "diagonal edge"
]

HEIGHT = 480
FEATURE_MIN = 0.25
FEATURE_MAX = 0.5


def img_resize(i):
    h, w, _ = i.shape
    return cv2.resize(i, (round(w * HEIGHT / h), HEIGHT))


"""
5 szablonów
s = 2
p = 2 -> 3x3 okno
lczba_cech = 5 * s**2 * (2*p - 1)**2 = 5 * 4 * 9 = 180
"""


def haar_indexes(s: int, p: int):
    h_indexes = []

    for t in range(len(HAAR_TEMPLATED)):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in range(-p + 1, p, 1):
                    for p_k in range(-p + 1, p, 1):
                        h_indexes.append(np.array([t, s_j, s_k, p_j, p_k]))

    return np.array(h_indexes)


def haar_coordinates(s: int, p: int, h_indexes: np.array):
    h_coords = []

    for t, s_j, s_k, p_j, p_k in h_indexes:
        f_h = FEATURE_MIN + s_j * (FEATURE_MAX - FEATURE_MIN) / (s - 1) if s > 1 else FEATURE_MIN
        f_w = FEATURE_MIN + s_k * (FEATURE_MAX - FEATURE_MIN) / (s - 1) if s > 1 else FEATURE_MIN
        shift_h = (1.0 - f_h) / (2 * p - 2) if p > 1 else 0.0
        shift_w = (1.0 - f_w) / (2 * p - 2) if p > 1 else 0.0
        pos_j = 0.5 + p_j * shift_h - 0.5 * f_h
        pos_k = 0.5 + p_k * shift_w - 0.5 * f_w

        single_hcoords = [np.array([pos_j, pos_k, f_h, f_w])]  # background of whole feature (useful later for feature computation)
        for white in HAAR_TEMPLATED[t]:
            single_hcoords.append(white * np.array([f_h, f_w, f_h, f_w]) + np.array([pos_j, pos_k, 0.0, 0.0]))
        h_coords.append(np.array(single_hcoords))

    return np.array(h_coords, dtype="object")


def draw_feature(image: np.array, j0: int, k0: int, hcoords_window: np.array):
    image_copy = image.copy()
    j, k, h, w = hcoords_window[0]  # first row, relative to window
    cv2.rectangle(image_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (0, 0, 0), cv2.FILLED)

    for white in hcoords_window[1:]:
        j, k, h, w = white
        cv2.rectangle(image_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (255, 255, 255), cv2.FILLED)

    return image_copy


def integral_image(image_gray: np.array):
    h, w = image_gray.shape
    ii = np.zeros(image_gray.shape, dtype='int32')
    ii_row = np.zeros(w, dtype='int32')

    for j in range(h):
        for k in range(w):
            ii_row[k] = image_gray[j, k]  # dodaj do sumy, która jest w bieżącym wierszu w pkt k
            if k > 0:  # jak było coś na lewo
                ii_row[k] += ii_row[k - 1]  # to doklej

            ii[j, k] = ii_row[k]
            if j > 0:
                ii[j, k] += ii[j - 1, k]

    return ii


def integral_image_cumsum(image_gray: np.array):
    return np.cumsum(np.cumsum(image_gray, axis=0), axis=1)


@jit(int32[:, :](uint8[:, :]), nopython=True, cache=True)
def integral_image_numba(image_gray: np.array):
    h, w = image_gray.shape
    ii = np.zeros(image_gray.shape, dtype='int32')
    ii_row = np.zeros(w, dtype='int32')

    for j in range(h):
        for k in range(w):
            ii_row[k] = image_gray[j, k]  # dodaj do sumy, która jest w bieżącym wierszu w pkt k
            if k > 0:  # jak było coś na lewo
                ii_row[k] += ii_row[k - 1]  # to doklej

            ii[j, k] = ii_row[k]
            if j > 0:
                ii[j, k] += ii[j - 1, k]

    return ii


@jit(int32(int32[:, :], int32, int32, int32, int32), nopython=True, cache=True)
def integral_image_delta(integral_image: np.array, j1: int, k1: int, j2: int, k2: int):
    # integral_image[j2, k2] - integral_image[j1 - 1, k2] - integral_image[j2, k1 - 1] + integral_image[j1 - 1, k1 - 1]
    delta = integral_image[j2, k2]

    if j1 > 0:
        delta -= integral_image[j1 - 1, k2]
    if k1 > 0:
        delta -= integral_image[j2, k1 - 1]
    if j1 > 0 and k1 > 0:
        delta += integral_image[j1 - 1, k1 - 1]

    return delta


if __name__ == '__main__':
    s = 2
    p = 2
    h_indexes = haar_indexes(s, p)
    print(len(h_indexes))

    i = cv2.imread(DATA_FOLDER + "000000.jpg")

    i_resized = img_resize(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)

    j0, k0 = 160, 280
    h = w = 128
    ii = integral_image_numba(i_gray)
    repetitions = 10000

    # measure of calculate time
    t1 = time.time()
    for repetition in range(repetitions):
        sum1 = integral_image_delta(ii, j0, k0, j0 + h - 1, k0 + w - 1)
    t2 = time.time()
    print(f'INTEGRAL_IMAGE_DELTA: {(t2 - t1) / repetitions} s')

    t1 = time.time()
    for repetition in range(repetitions):
        sum2 = np.sum(i_gray[j0: j0 + h, k0: k0 + w])
    t2 = time.time()
    print(f'STANDART NP>NUM TIME: {(t2 - t1) / repetitions} s')
    print(f'{sum1} vs {sum2}')

    # my own rectangle on image
    # j0, k0 = 160, 280
    # h = w = 64
    # cv2.rectangle(i_resized, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255), 1)
    # cv2.imshow("TEST IMAGE", i_resized)
    # cv2.waitKey()
    #
    # h_coords = haar_coordinates(s, p, h_indexes)
    # for i, c in zip(h_indexes, h_coords):
    #     # print(f'{i} -> {c}')
    #     h_coords_window = (c * h).astype('int32')  # np.array([np.array(c[q] * h).astype('int32') for q in range(c.shape[0])])
    #     image_with_feature = draw_feature(i_resized, j0, k0, h_coords_window)
    #     image_temp = cv2.addWeighted(i_resized, 0.5, image_with_feature, 0.5, 0.0)
    #
    #     cv2.imshow('TEST IMAGE', image_temp)
    #     cv2.waitKey()
    #
    #     print(f'INDEX: {i}')
    #     print(f'HCOORDS:\n {c}')
    #     print(f'HCOORDS_WINDOW:\n {h_coords_window}')
    #     print('--\n')
