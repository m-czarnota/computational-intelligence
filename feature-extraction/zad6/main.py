from typing import Tuple

import numpy as np
import pandas as pd
import os
import cv2
from sklearn.neural_network import MLPClassifier

IMAGES_DIR = './images'


def read_images() -> dict:
    images = {'black': [], 'contour': []}

    dirs = [x[0] for x in os.walk(IMAGES_DIR)]
    black_and_white_dirs = get_list_of_dirs(dirs, 'black_and_white')

    for dir_index, black_and_white_dir in enumerate(black_and_white_dirs):
        filenames = [x[2] for x in os.walk(black_and_white_dir)][0]
        images['black'].append([])
        images['contour'].append([])

        for filename_iter, filename in enumerate(filenames):
            image = cv2.imread(f'{black_and_white_dir}/{filename}')
            images['black'][dir_index].append(convert_image_to_black(image))
            images['contour'][dir_index].append(convert_image_to_contour(image))

    return images


def select_representatives_numbers(images: dict) -> list:
    representative_images_number: dict = {
        'gambles-quail': 6,
        'glossy-ibis': 5,
        'greator-sage-grous': 3,
        'hooded-merganser': 5,
        'indian-vulture': 4,
        'jabiru': 1,
        'king-eider': 2,
        'long-eared-owl': 4,
        'tit-mouse': 4,
        'touchan': 3}  # from 1

    return list(map(lambda number: number - 1, representative_images_number.values()))


def train_test_split(images: dict, representative_numbers: list) -> Tuple:
    train_images = {'black': [], 'contour': []}
    test_images = {'black': [], 'contour': []}

    for number_iter, number in enumerate(representative_numbers):
        train_images['black'].append(images['black'][number_iter][number])
        train_images['contour'].append(images['contour'][number_iter][number])

        test_images['black'].append([])
        test_images['contour'].append([])

        for image_index in range(len(images['black'])):
            if image_index == number:
                continue

            test_images['black'][number_iter].append(images['black'][number_iter][image_index])
            test_images['contour'][number_iter].append(images['contour'][number_iter][image_index])

    return train_images, test_images


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color'):
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


def area_descriptor(image: np.array) -> float:
    return np.array(list(zip(*np.where(image == 0)))).shape[0]


def perimeter_descriptor(contour_image: np.array) -> float:
    return area_descriptor(contour_image)


def roundness_descriptor(image: np.array) -> float:
    return perimeter_descriptor(image) ** 2 / (4 * np.pi * area_descriptor(image))


def compactness_descriptor(image: np.array) -> float:
    return perimeter_descriptor(image) ** 2 / area_descriptor(image)


def eccentricity_descriptor(image: np.array) -> float:
    boundary = {'x_min': image.shape[0], 'x_max': 0, 'y_min': image.shape[1], 'y_max': 0}
    
    for x, row in enumerate(image):
        for y, val in enumerate(row):
            if val != 0:
                continue
                
            if x < boundary['x_min']:
                boundary['x_min'] = x
                
            if x > boundary['x_max']:
                boundary['x_max'] = x

            if y < boundary['y_min']:
                boundary['y_min'] = y

            if y > boundary['y_max']:
                boundary['y_max'] = y

    rectangle_boundary = [boundary['x_max'] - boundary['x_min'], boundary['y_max'] - boundary['y_min']]

    return np.max(rectangle_boundary) / np.min(rectangle_boundary)


def calc_distance(i, j):
    return np.sqrt(np.abs(i ** 2 - j ** 2))


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
        image_type = 'black' if descriptor.__name__ != perimeter_descriptor.__name__ else 'contour'

        descriptor_train_result = [descriptor(image) for image in train_images[image_type]]
        train_results[descriptor.__name__] = descriptor_train_result

        descriptor_test_result = []
        for test_class_images in test_images[image_type]:
            test_result_class = [descriptor(test_image) for test_image in test_class_images]
            descriptor_test_result.append(test_result_class)

        test_results[descriptor.__name__] = descriptor_test_result

    return train_results, test_results


if __name__ == '__main__':
    descriptors = [area_descriptor, perimeter_descriptor, roundness_descriptor, compactness_descriptor, eccentricity_descriptor]

    images = read_images()
    representatives_numbers = select_representatives_numbers(images)
    train_images, test_images = train_test_split(images, representatives_numbers)

    train_results, test_results = calc_descriptors_for_images(train_images, test_images)

    results = {}

    for descriptor_name, descriptor_train_class_results in train_results.items():
        indexes = np.arange(len(descriptor_train_class_results))
        descriptor_results = np.empty((indexes.size - 1, indexes.size))

        for class_iter, train_class_result in enumerate(descriptor_train_class_results):
            for test_class_result_iter, test_class_result in enumerate(test_results[descriptor_name][class_iter]):
                descriptor_results[test_class_result_iter, class_iter] = calc_distance(train_class_result, test_class_result)

        results[descriptor_name] = np.argmin(descriptor_results, axis=1)

    test = 2


    # classifiers_by_descriptor = [MLPClassifier(max_iter=1000) for _ in descriptors]
    # scores = {descriptor.__name__: 0 for descriptor in descriptors}
    # for clf, (descriptor_name, results) in zip(classifiers_by_descriptor, descriptor_results.items()):
    #     clf.fit(results['results'], results['class'])
    #     scores[descriptor_name] = clf.score(results['results'], results['class'])
    #
    # print(scores)
    #
    # scores = {descriptor.__name__: np.zeros(10) for descriptor in descriptors}
    # for descriptor_name, results_for_descriptor in scores.items():
    #     for label in range(results_for_descriptor.size):
    #         clf = MLPClassifier(max_iter=1000)
    #         clf.fit(descriptor_results[descriptor_name]['results'], descriptor_results[descriptor_name]['class'])
    #         scores[descriptor_name][label] = clf.score(descriptor_results[descriptor_name]['results'], descriptor_results[descriptor_name]['class'])
    #
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(pd.DataFrame(scores).to_markdown())
