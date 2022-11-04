import cv2
import numpy as np
import time
from numba import jit, uint8, int32, int16
import pickle
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from src.RealBoostBins import RealBoostBins

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

DETECT_SCALES = 4
DETECT_WINDOW_HEIGHT_MIN = 64
DETECT_WINDOW_WIDTH_MIN = 64
DETECT_WINDOW_GROWTH = 1.25  # increase window about 25%
DETECT_WINDOW_JUMP = 0.1
DETECT_THRESHOLD = 0.15


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

        single_hcoords = [
            np.array([pos_j, pos_k, f_h, f_w])]  # background of whole feature (useful later for feature computation)
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


@jit(int16(int32[:, :], int32, int32, int32[:, :]), nopython=True, cache=True)
def haar_feature(integral_image: np.array, j0: int, k0: int, haar_coords_window):
    """
    (j0, k0) - window top left corner
    haar_coords_window - coordinated of single feature (in pixels) / współrzędne cechy haara w pikselach

    można by trzymać wartość cechy haara w double, ale to będzie rozrzutność pamięciowa, bo double ma 8 bitów
    double będzie miał nikły wpłw na wartość klasyfikatora
    """
    j, k, h, w = haar_coords_window[0]  # whole feature background - feature tutaj to cecha, cecha haara
    total_area = h * w
    j1 = j0 + j
    k1 = k0 + k

    total_intensity = integral_image_delta(integral_image, j1, k1, j1 + h - 1,
                                           k1 + w - 1)  # suma jasności pikseli pod całym ty prostokątem, który pokrywa cecha
    white_area = 0  # pole białych jest 0
    white_intensity = 0  # intensywność białych jest 0

    for white in haar_coords_window[1:]:
        j, k, h, w = white
        white_area += h * w  # szerokość * wysokość

        j1 = j0 + j
        k1 = k0 + k
        white_intensity += integral_image_delta(integral_image, j1, k1, j1 + h - 1, k1 + w - 1)

    black_area = total_area - white_area  # powierzchnia pod czarnymi
    black_intensity = total_intensity - white_intensity

    return np.int16(white_intensity / white_area - black_intensity / black_area)


def haar_features(integral_image: np.array, j0: int, k0: int, haar_coords_window_subset, n, feature_indexes=None):
    """
    haar_coords_window_subset - przy generowaniu wielu ten subset nie będzie podzbiorem
    jak będziemy generowali zbiór uczący to feature_indexes będzie None
    """
    features = np.zeros(n, dtype='int16')
    if feature_indexes is None:
        feature_indexes = np.arange(n)

    for i, feature_index in enumerate(feature_indexes):
        features[feature_index] = haar_feature(integral_image, j0, k0, haar_coords_window_subset[i])

    return features


def iou(coords_1, coords_2):
    j11, k11, j12, k12 = coords_1
    j21, k21, j22, k22 = coords_2
    dj = np.min([j12, j22]) - np.max([j21, j11]) + 1
    if dj <= 0:
        return 0.0
    dk = np.min([k12, k22]) - np.max([k21, k11]) + 1
    if dk <= 0:
        return 0.0
    i = dj * dk
    u = (j12 - j11 + 1) * (k12 - k11 + 1) + (j22 - j21 + 1) * (k22 - k21 + 1) - i
    return i / u


def fddb_read_single_fold(path_root, path_fold_relative, n_negs_per_img, hfs_coords: np.array, n: int, verbose: bool = False, fold_title: str = ""):
    np.random.seed(1)

    # settings for sampling negatives
    w_relative_min = 0.1
    w_relative_max = 0.35
    w_relative_spread = w_relative_max - w_relative_min
    neg_max_iou = 0.5

    X_list = []
    y_list = []

    f = open(path_root + path_fold_relative, "r")
    line = f.readline().strip()
    n_img = 0
    n_faces = 0
    counter = 0

    while line != "":
        file_name = path_root + line + ".jpg"
        log_line = str(counter) + ": [" + file_name + "]"
        if fold_title != "":
            log_line += " [" + fold_title + "]"
        print(log_line)
        counter += 1

        i0 = cv2.imread(file_name)
        i = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
        ii = integral_image(i)
        n_img += 1
        n_img_faces = int(f.readline())
        img_faces_coords = []

        for z in range(n_img_faces):
            r_major, r_minor, angle, center_x, center_y, dummy_one = list(map(float, f.readline().strip().split()))
            w = int(1.5 * r_major)
            j0 = int(center_y - w / 2)
            k0 = int(center_x - w / 2)
            img_face_coords = np.array([j0, k0, j0 + w - 1, k0 + w - 1])

            if j0 < 0 or k0 < 0 or j0 + w - 1 >= i.shape[0] or k0 + w - 1 >= i.shape[1]:
                if verbose:
                    print("WINDOW " + str(img_face_coords) + " OUT OF BOUNDS. [IGNORED]")
                continue

            if w / ii.shape[0] < 0.075:  # min relative size of positive window (smaller may lead to division by zero when white regions in haar features have no area)
                if verbose:
                    print("WINDOW " + str(img_face_coords) + " TOO SMALL. [IGNORED]")
                continue

            n_faces += 1
            img_faces_coords.append(img_face_coords)

            if verbose:
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + w - 1)
                cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)
                cv2.imshow("FDDB", i0)

            hfs_coords_window = w * hfs_coords
            hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window)), dtype='object')
            feats = haar_features(ii, j0, k0, hfs_coords_window, n)

            if verbose:
                print("POSITIVE WINDOW " + str(img_face_coords) + " ACCEPTED. FEATURES: " + str(feats) + ".")
                cv2.waitKey(0)

            X_list.append(feats)
            y_list.append(1)

        for z in range(n_negs_per_img):
            while True:
                w = int((np.random.random() * w_relative_spread + w_relative_min) * i.shape[0])
                j0 = int(np.random.random() * (i.shape[0] - w + 1))
                k0 = int(np.random.random() * (i.shape[1] - w + 1))

                patch = np.array([j0, k0, j0 + w - 1, k0 + w - 1])
                ious = list(map(lambda ifc: iou(patch, ifc), img_faces_coords))
                max_iou = max(ious) if len(ious) > 0 else 0.0

                if max_iou < neg_max_iou:
                    hfs_coords_window = w * hfs_coords
                    hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window)), dtype='object')
                    feats = haar_features(ii, j0, k0, hfs_coords_window, n)

                    X_list.append(feats)
                    y_list.append(-1)

                    if verbose:
                        print("NEGATIVE WINDOW " + str(patch) + " ACCEPTED. FEATURES: " + str(feats) + ".")
                        p1 = (k0, j0)
                        p2 = (k0 + w - 1, j0 + w - 1)
                        cv2.rectangle(i0, p1, p2, (0, 255, 0), 1)

                    break
                else:
                    if verbose:
                        print("NEGATIVE WINDOW " + str(patch) + " IGNORED. [MAX IOU: " + str(max_iou) + "]")
                        p1 = (k0, j0)
                        p2 = (k0 + w - 1, j0 + w - 1)
                        cv2.rectangle(i0, p1, p2, (255, 255, 0), 1)

        if verbose:
            cv2.imshow("FDDB", i0)
            cv2.waitKey(0)

        line = f.readline().strip()

    print("IMAGES IN THIS FOLD: " + str(n_img) + ".")
    print("ACCEPTED FACES IN THIS FOLD: " + str(n_faces) + ".")

    f.close()
    X = np.stack(X_list)
    y = np.stack(y_list)

    return X, y


def fddb_data(path_fddb_root, hfs_coords, n_negs_per_img, n):
    n_negs_per_img = n_negs_per_img

    fold_paths_train = [
        "FDDB-folds/FDDB-fold-01-ellipseList.txt",
        "FDDB-folds/FDDB-fold-02-ellipseList.txt",
        "FDDB-folds/FDDB-fold-03-ellipseList.txt",
        "FDDB-folds/FDDB-fold-04-ellipseList.txt",
        "FDDB-folds/FDDB-fold-05-ellipseList.txt",
        "FDDB-folds/FDDB-fold-06-ellipseList.txt",
        "FDDB-folds/FDDB-fold-07-ellipseList.txt",
        "FDDB-folds/FDDB-fold-08-ellipseList.txt",
        "FDDB-folds/FDDB-fold-09-ellipseList.txt"
    ]
    X_train = None
    y_train = None

    for index, fold_path in enumerate(fold_paths_train):
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + "...")
        t1 = time.time()
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, n_negs_per_img, hfs_coords, n, verbose=False,
                                     fold_title=fold_path)
        t2 = time.time()
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + " DONE IN " + str(
            t2 - t1) + " s.")
        print("---")

        if X_train is None:
            X_train = X
            y_train = y
        else:
            X_train = np.r_[X_train, X]
            y_train = np.r_[y_train, y]

    fold_paths_test = [
        "FDDB-folds/FDDB-fold-10-ellipseList.txt",
    ]
    X_test = None
    y_test = None

    for index, fold_path in enumerate(fold_paths_test):
        print("PROCESSING TEST FOLD " + str(index + 1) + "/" + str(len(fold_paths_test)) + "...")
        t1 = time.time()
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, n_negs_per_img, hfs_coords, n, fold_title=fold_path)
        t2 = time.time()
        print("PROCESSING TEST FOLD " + str(index + 1) + "/" + str(len(fold_paths_test)) + " DONE IN " + str(
            t2 - t1) + " s.")
        print("---")

        if X_test is None:
            X_test = X
            y_test = y
        else:
            X_test = np.r_[X_test, X]
            y_test = np.r_[y_test, y]

    print("TRAIN DATA SHAPE: " + str(X_train.shape))
    print("TEST DATA SHAPE: " + str(X_test.shape))

    return X_train, y_train, X_test, y_test


def pickle_all(fname, some_list):
    print("PICKLE...")
    t1 = time.time()
    f = open(fname, "wb+")
    pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time.time()
    print("PICKLE DONE. [TIME: " + str(t2 - t1) + " s.]")


def unpickle_all(fname):
    print("UNPICKLE...")
    t1 = time.time()
    f = open(fname, "rb")
    some_list = pickle.load(f)
    f.close()
    t2 = time.time()
    print("UNPICKLE DONE. [TIME: " + str(t2 - t1) + " s.]")

    return some_list


def time_of_comparisons_delta():
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
    print(f'STANDARD NP>NUM TIME: {(t2 - t1) / repetitions} s')
    print(f'{sum1} vs {sum2}')


def demo_of_features():
    # my own rectangle on image
    j0, k0 = 160, 280
    h = w = 64
    cv2.rectangle(i_resized, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255), 1)
    cv2.imshow("TEST IMAGE", i_resized)
    cv2.waitKey()

    h_coords = haar_coordinates(s, p, h_indexes)
    for index, (i, c) in enumerate(zip(h_indexes[selected_feature_indexes], h_coords[selected_feature_indexes])):
        # print(f'{i} -> {c}')
        """
        h_coords_window to jedno pojedyczne okienko z h_coords
        """
        h_coords_window = (c * h).astype(
            'int32')  # np.array([np.array(c[q] * h).astype('int32') for q in range(c.shape[0])])
        image_with_feature = draw_feature(i_resized, j0, k0, h_coords_window)
        image_temp = cv2.addWeighted(i_resized, 0.5, image_with_feature, 0.5, 0.0)

        print(f'INDEX: {i}')
        print(f'FEATURE_INDEX: {selected_feature_indexes[index]}')
        print(f'HCOORDS:\n {c}')
        print(f'HCOORDS_WINDOW:\n {h_coords_window}')
        print(f'HAAR FEATURE:\n {haar_feature(ii, j0, k0, h_coords_window)}')
        print('--\n')

        cv2.imshow('TEST IMAGE', image_temp)
        plt.hist(X_train[indexed_positive_train, selected_feature_indexes[index]], color='r', density=True, alpha=0.5,
                 bins=20, label='positives')
        plt.hist(X_train[indexed_negative_train, selected_feature_indexes[index]], color='b', density=True, alpha=0.5,
                 bins=20, label='negatives')
        plt.legend()
        plt.show()
        cv2.waitKey()


def haar_features_demo():
    h_coords = haar_coordinates(s, p, h_indexes)
    # h_coords_window_subset = (h_coords * h).astype('int32')  # error
    h_coords_window_subset = np.array([np.array(h_coords[q] * h).astype('int32') for q in range(h_coords.shape[0])], dtype='object')
    t1 = time.time()
    features = haar_features(ii, j0, k0, h_coords_window_subset, n)
    t2 = time.time()
    print(features, f'time: {t2 - t1}')


def detect(classifier: AdaBoostClassifier, image: np.array, h_coords, n, feature_indexes=None, preprocess: bool = True, verbose: bool = False):
    t1 = time.time()

    print(f'[IMAGE SHAPE: {image.shape}]')
    if preprocess:
        t1_preprocess = time.time()

        i_resized = img_resize(i)
        i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)

        t2_preprocess = time.time()
        print(f'PREPROCESS DONE. [TIME: {t2_preprocess - t1_preprocess}s]')
    else:
        i_gray = image

    t1_ii = time.time()
    ii = integral_image(i_gray)
    t2_ii = time.time()
    print(f'INTEGRAL IMAGE DONE. [TIME: {t2_ii - t1_ii}s]')

    H, W = i_gray.shape
    print(f'IMAGE SHAPE: {i_gray.shape}')

    windows = []
    h_coords_window_subsets = []
    h_coords_subset = h_coords[feature_indexes] if feature_indexes is not None else h_coords
    windows_count = 0

    t1_raw_loops = time.time()
    for scale in range(DETECT_SCALES):
        h = np.int32(np.round(DETECT_WINDOW_HEIGHT_MIN * DETECT_WINDOW_GROWTH ** scale))
        w = np.int32(np.round(DETECT_WINDOW_WIDTH_MIN * DETECT_WINDOW_GROWTH ** scale))

        # jumps
        dj = np.int32(np.round(h * DETECT_WINDOW_JUMP))
        dk = np.int32(np.round(w * DETECT_WINDOW_JUMP))

        h_reminder_half = ((H - h) % dj) // 2
        w_reminder_half = ((W - w) % dk) // 2

        h_coords_window_subsets.append(np.array([np.array(h_coords[q] * h).astype("int32") for q in range(h_coords_subset.shape[0])]))

        for j in np.arange(h_reminder_half, H - h + 1, dj):
            for k in np.arange(w_reminder_half, W - w + 1, dk):
                windows_count += 1
                windows.append([scale, j, k, h, w])

    t2_raw_loops = time.time()
    print(f'RAW LOOPS DONE> [TIME: {t2_raw_loops - t1_raw_loops}s, WINDOWS TO CHECK: {windows_count}]')

    t1_main_loop = time.time()
    detections = []
    progress_check = int(np.round(0.1 * windows_count))

    for window_index, (scale, j, k, h, w) in enumerate(windows):
        if window_index % progress_check == 0:
            print(f'PROGRESS: {window_index / windows_count:.2}')

        features = haar_features(ii, j, k, h_coords_window_subsets[scale], n, feature_indexes=feature_indexes)
        response = classifier.decision_function(np.array([features]))

        if response > DETECT_THRESHOLD:
            detections.append(np.array([j, k, h, w]))

    t2_main_loop = time.time()
    print(f'MAIN LOOP DONE. [TIME: {t2_main_loop - t1_main_loop}s]')

    # for scale in range(DETECT_SCALES):
    #     h = np.int32(np.round(DETECT_WINDOW_HEIGHT_MIN * DETECT_WINDOW_GROWTH ** scale))
    #     w = np.int32(np.round(DETECT_WINDOW_WIDTH_MIN * DETECT_WINDOW_GROWTH ** scale))
    #
    #     # jumps
    #     dj = np.int32(np.round(h * DETECT_WINDOW_JUMP))
    #     dk = np.int32(np.round(w * DETECT_WINDOW_JUMP))
    #
    #     if verbose is True:
    #         print('--------------------------------------------------------')
    #         print(f'SCALE {scale} -> h: {h}, w: {w}, dj: {dj}, dk: {dk}')
    #
    #     h_reminder_half = ((H - h) % dj) // 2
    #     w_reminder_half = ((W - w) % dk) // 2
    #
    #     h_coords_window_subset = np.array([np.array(h_coords_subset[q] * h).astype('int32') for q in range(h_coords_subset.shape[0])], dtype='object')
    #
    #     for j in np.arange(h_reminder_half, H - h + 1, dj):
    #         if verbose is True:
    #             print(f'j: {j}, windows so far: {windows_count}')
    #
    #         for k in np.arange(w_reminder_half, W - w + 1, dk):
    #             features = haar_features(ii, j, k, h_coords_window_subset, n, feature_indexes=feature_indexes)
    #             response = classifier.decision_function(np.array([features]))
    #
    #             if response > DETECT_THRESHOLD:
    #                 detections.append(np.array([j, k, h, w]))
    #
    #             windows_count += 1

    t2 = time.time()
    print(f'DETECT DONE. [TIME: {t2 - t1}s, WINDOWS CHECKED: {windows_count}]')

    return detections


def get_metrics(classifier, x, y):
    # acc_train = np.mean(classifier.predict(X_train) == y_train)
    acc_train = classifier.score(x, y)
    sensitivity_train = classifier.score(x[indexed_positive_train], y[indexed_positive_train])
    false_alarm_rate_train = 1.0 - classifier.score(x[indexed_negative_train], y[indexed_negative_train])

    return acc_train, sensitivity_train, false_alarm_rate_train


if __name__ == '__main__':
    s = 3
    p = 4
    n = len(HAAR_TEMPLATED) * s ** 2 * (2 * p - 1) ** 2
    T = 8  # number of boosting rounds
    B = 8  # number fo bins (buckets)
    random_seed = 1

    DATA_NAME = f'face_n_{n}_s_{s}_p_{p}.bin'
    CLFS_NAME = f'face_n_{n}_s_{s}_p_{p}_T_{T}_ada.bin'
    print(f's: {s}, p: {p}, n: {n}')

    h_indexes = haar_indexes(s, p)
    haar_coords = haar_coordinates(s, p, h_indexes)
    print(len(h_indexes), haar_coords.shape[0])

    i = cv2.imread(DATA_FOLDER + "000000.jpg")

    i_resized = img_resize(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)

    j0, k0 = 160, 280
    h = w = 128
    ii = integral_image(i_gray)
    h_coords = haar_coordinates(s, p, h_indexes)

    # time_of_comparisons_delta()
    # demo_of_features()

    t1 = time.time()
    # X_train, y_train, X_test, y_test = fddb_data(DATA_FOLDER, h_coords, 10, n)
    # pickle_all(DATA_FOLDER + DATA_NAME, [X_train, y_train, X_test, y_test])
    X_train, y_train, X_test, y_test = unpickle_all(DATA_FOLDER + DATA_NAME)
    t2 = time.time()
    indexed_positive_train = y_train == 1
    indexed_negative_train = y_train == -1
    indexed_positive_test = y_test == 1
    indexed_negative_test = y_test == -1

    # --- ADA BOOST ---
    # t1 = time.time()
    # classifier = AdaBoostClassifier(n_estimators=T, algorithm='SAMME', random_state=random_seed)
    # classifier.fit(X_train, y_train)
    # t2 = time.time()
    # print(f'fit time: {t2 - t1}s')
    # pickle_all(CLFS_FOLDER + CLFS_NAME, [classifier])
    # [classifier] = unpickle_all(CLFS_FOLDER + CLFS_NAME)
    # selected_feature_indexes = np.where(classifier.feature_importances_ > 0)[0]

    # acc_train, sensitivity_train, false_alarm_rate_train = get_metrics(classifier, X_train, y_train)
    # print(f'ACC TRAIN: {acc_train}, SENS TRAIN: {sensitivity_train}, FAR TRAIN: {false_alarm_rate_train}')

    # --- REAL BOOST ---
    clf = RealBoostBins(t=T, b=B)
    clf.fit(X_train, y_train)
    print(clf.logits_)

    # --- ACCURACY MEASURES ---
    # acc_test = classifier.score(X_test, y_test)
    # sensitivity_test = classifier.score(X_test[indexed_positive_test], y_test[indexed_positive_test])
    # false_alarm_rate_test = 1.0 - classifier.score(X_test[indexed_negative_test], y_test[indexed_negative_test])
    # print(f'ACC TEST: {acc_test}, SENS TEST: {sensitivity_test}, FAR TEST: {false_alarm_rate_test}')

    # detections = detect(classifier, i, h_coords, n, selected_feature_indexes, preprocess=True, verbose=True)
    #
    # for j0, k0, h, w in detections:
    #     cv2.rectangle(i_resized, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255))
    # cv2.imshow("TEST IMAGE", i_resized)
    # cv2.waitKey()

    """
    zad domowe:
    odkomentować resztę plików w fddb_data, zwiększyć rozmiar okienka s=3 i p=4, gdzie będzie n=2225, i s=5 oraz p=5
    zrzucić wyniki do plików
    zwiększyć n z 3 na 10 przy fddb_data
    """
