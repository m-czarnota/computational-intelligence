import copy
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
        'gambles-quail': [6, 9, 10, 2, 1],
        'glossy-ibis': [10, 9, 8, 2, 3],
        'greator-sage-grous': [5, 6, 9, 10, 8],
        'hooded-merganser': [1, 2, 5, 9, 4],
        'indian-vulture': [4, 2, 6, 8, 9],
        'jabiru': [1, 6, 7, 10, 5],
        'king-eider': [8, 2, 5, 7, 6],
        'long-eared-owl': [4, 3, 1, 8, 9],
        'tit-mouse': [8, 5, 6, 10, 1],
        'touchan': [6, 2, 1, 10, 9],
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


def calc_descriptors_for_images(train_images: list, test_images: list) -> Tuple:
    train_results = {descriptor.__class__.__name__: [] for descriptor in descriptors}
    test_results = {descriptor.__class__.__name__: [] for descriptor in descriptors}

    for descriptor in descriptors:
        descriptor_train_result = []
        for train_class_images in train_images:
            train_result_class = [copy.deepcopy(descriptor).descript_image(train_image) for train_image in train_class_images]
            descriptor_train_result.append(train_result_class)
        train_results[descriptor.__class__.__name__] = descriptor_train_result

        descriptor_test_result = []
        for test_class_images in test_images:
            test_result_class = [copy.deepcopy(descriptor).descript_image(test_image) for test_image in test_class_images]
            descriptor_test_result.append(test_result_class)
        test_results[descriptor.__class__.__name__] = descriptor_test_result

    return train_results, test_results


def test_descriptors():
    test_image = test_images[0][4]

    cdp = CDP()
    cdp.descript_image(test_image)
    print(cdp.distances)

    plt.figure()
    plt.plot(cdp.distances)
    plt.show()

    log_pol = LogPol()
    log_pol.descript_image(test_image)
    print(log_pol.p.shape, log_pol.w.shape)

    pdh = CentroidPDH(points_count=200)
    pdh.descript_image(test_image)
    print(pdh.h.shape)


def display_results(results: dict):
    results_view = pd.DataFrame()
    scores_by_descriptor = {}

    labels = range(10)
    score_by_class = {label: 0 for label in labels}

    for descriptor_name, descriptor_results in results.items():
        # print(f'------------ {descriptor_name} ------------')
        # print(descriptor_results.to_markdown())

        score = descriptor_results['score'].sum() / descriptor_results.shape[0] * 100
        scores_by_descriptor[descriptor_name] = f'{score:.4f}%'

        for label in labels:
            class_data = descriptor_results[descriptor_results['real'] == label]
            class_data_count = class_data['score'].count()
            score_by_class[label] = class_data['score'].sum() / class_data_count

        scores_df = pd.DataFrame(np.array([[*score_by_class.keys()], [*score_by_class.values()]]).T,
                                 columns=['class', 'score'])
        scores_df['class'] = scores_df['class'].map(lambda x: x + 1)
        scores_df['score'] = scores_df['score'].map(lambda x: f'{(x * 100):0.4f}%')

        results_view['class'] = scores_df['class']
        results_view[descriptor_name] = scores_df['score']

    scores_by_descriptor = pd.DataFrame(pd.Series(scores_by_descriptor), columns=['score'])
    print(scores_by_descriptor.to_markdown())

    descriptor_values = results_view[list(map(lambda desc: desc.__class__.__name__, descriptors))].applymap(
        lambda val: float(val.replace('%', '')))
    results_view['score'] = descriptor_values.sum(axis=1) / len(descriptors)
    results_view['score'] = results_view['score'].map(lambda x: f'{x:0.4f}%')
    print(results_view.to_markdown())


if __name__ == '__main__':
    descriptors = [CDP(200), LogPol(200), CentroidPDH(5)]
    descriptor_names = [descriptor.__class__.__name__ for descriptor in descriptors]

    images = read_images()
    representatives_numbers = select_representatives_numbers()
    train_images, test_images = train_test_split(images, representatives_numbers)

    train_results, test_results = calc_descriptors_for_images(train_images, test_images)
    results = {}
    results_view = pd.DataFrame()

    # test_descriptors()

    for descriptor_name in descriptor_names:
        train_descriptors = train_results[descriptor_name]
        test_descriptors = test_results[descriptor_name]
        descriptor_results = pd.DataFrame(columns=['predicted', 'real', 'score'])

        for class_iter, test_class_descriptors in enumerate(test_descriptors):
            for test_descriptor in test_class_descriptors:
                train_descriptor_classes_distances_to_test = []
                for train_class_descriptors in train_descriptors:
                    train_descriptor_classes_distances_to_test.append(np.min([train_descriptor.calc_distance_to_other_descriptor(test_descriptor) for train_descriptor in train_class_descriptors]))

                predicted_class = np.argmin(train_descriptor_classes_distances_to_test)
                row = pd.Series({
                    'predicted': predicted_class,
                    'real': class_iter,
                    'score': int(predicted_class == class_iter),
                })
                descriptor_results = pd.concat([descriptor_results, row.to_frame().T], ignore_index=True)

        results[descriptor_name] = pd.DataFrame(descriptor_results)

    display_results(results)
