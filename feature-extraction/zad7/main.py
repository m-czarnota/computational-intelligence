import os
from typing import Tuple
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.descriptor.CDP import CDP
from src.descriptor.CentroidPDH import CentroidPDH
from src.descriptor.LogPol import LogPol

IMAGES_DIR = './images'


def read_images() -> list:
    images = []

    dirs = [x[0] for x in os.walk(IMAGES_DIR)]
    black_and_white_dirs = get_list_of_dirs(dirs, 'black_and_white')

    for dir_index, black_and_white_dir in enumerate(black_and_white_dirs):
        filenames = [x[2] for x in os.walk(black_and_white_dir)][0]
        images.append([])

        for filename_iter, filename in enumerate(filenames):
            image = cv2.imread(f'{black_and_white_dir}/{filename}')
            images[dir_index].append(convert_image_to_contour(image))

    return images


def select_representatives_numbers() -> list:
    representative_images_number: dict = {
        'gambles-quail': [6],
        'glossy-ibis': [10],
        'greator-sage-grous': [5],
        'hooded-merganser': [1],
        'indian-vulture': [4],
        'jabiru': [1],
        'king-eider': [8],
        'long-eared-owl': [4],
        'tit-mouse': [8],
        'touchan': [6],
    }  # from 1

    return [[number - 1 for number in numbers] for numbers in representative_images_number.values()]


def train_test_split(images: list, representative_numbers: list) -> Tuple:
    train_images = []
    test_images = []

    for numbers_iter, numbers in enumerate(representative_numbers):
        train_images.append([])
        test_images.append([])

        for index in range(len(images)):
            selected_image = images[numbers_iter][index]

            if index in numbers:
                train_images[numbers_iter].append(selected_image)
            else:
                test_images[numbers_iter].append(selected_image)

    return train_images, test_images


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color'):
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


def convert_image_to_black(image: np.array):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray_inv = 255 - image_gray
    tresh, image_black_and_white = cv2.threshold(image_gray_inv, 0, 255, cv2.THRESH_BINARY_INV)

    return image_black_and_white


def convert_image_to_contour(image: np.array):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh, image_black_and_white = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    image_contour = image_black_and_white.copy()
    image_contour[:] = 255

    contours, hierarchy = cv2.findContours(image_black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=image_contour, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=1)

    return image_contour


def calc_descriptors_for_images(train_images: dict, test_images: dict) -> Tuple:
    train_results = {descriptor.__name__: [] for descriptor in descriptors}
    test_results = {descriptor.__name__: [] for descriptor in descriptors}

    for descriptor in descriptors:
        descriptor_train_result = []
        for train_class_images in train_images:
            train_result_class = [descriptor.descript_image(train_image) for train_image in train_class_images]
            descriptor_train_result.append(train_result_class)
        train_results[descriptor.__name__] = descriptor_train_result

        descriptor_test_result = []
        for test_class_images in test_images:
            test_result_class = [descriptor.descript_image(test_image) for test_image in test_class_images]
            descriptor_test_result.append(test_result_class)
        test_results[descriptor.__name__] = descriptor_test_result

    return train_results, test_results


if __name__ == '__main__':
    descriptors = []

    images = read_images()
    representatives_numbers = select_representatives_numbers()
    train_images, test_images = train_test_split(images, representatives_numbers)

    train_results, test_results = calc_descriptors_for_images(train_images, test_images)
    results = {}
    results_view = pd.DataFrame()

    cdp = CDP()
    cdp.descript_image(test_images[0][4])
    print(cdp.distances)

    plt.figure()
    plt.plot(cdp.distances)
    plt.show()

    # log_pol = LogPol()
    # log_pol.descript_image(test_images[0][4])
    # print(log_pol.p.shape, log_pol.w.shape)

    # pdh = CentroidPDH(points_count=200)
    # pdh.descript_image(test_images[0][4])
    # print(pdh.h.shape)

    # for descriptor_name, train_descriptor_results in train_results.items():
    #     descriptor_results = pd.DataFrame(columns=['predicted', 'real', 'score'])
    #
    #     for class_iter, test_descriptor_results in enumerate(test_results[descriptor_name]):
    #         for test_descriptor_result in test_descriptor_results:
    #             predicted_class = predict_based_on_distance(test_descriptor_result, train_descriptor_results)
    #
    #             row = pd.Series(
    #                 {'predicted': predicted_class, 'real': class_iter, 'score': int(predicted_class == class_iter)})
    #             descriptor_results = pd.concat([descriptor_results, row.to_frame().T], ignore_index=True)
    #
    #     results[descriptor_name] = descriptor_results
