import numpy as np
import pandas as pd
import os
import cv2
from sklearn.neural_network import MLPClassifier

IMAGES_DIR = './images'


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color'):
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


def area_descriptor(image: np.array):
    return np.array(list(zip(*np.where(image == 0)))).shape[0]


def perimeter_descriptor(image: np.array):
    thresh, image_black_and_white = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    image_contour = image_black_and_white.copy()
    image_contour[:] = 255

    contours, hierarchy = cv2.findContours(image_black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=image_contour, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=1)

    # cv2.imshow('contours', image_contour)

    return area_descriptor(image_contour)


def roundness_descriptor(image: np.array):
    return perimeter_descriptor(image) ** 2 / (4 * np.pi * area_descriptor(image))


def compactness_descriptor(image: np.array):
    return perimeter_descriptor(image) ** 2 / area_descriptor(image)


def eccentricity_descriptor(image: np.array):
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


if __name__ == '__main__':
    descriptors = [area_descriptor, perimeter_descriptor, roundness_descriptor, compactness_descriptor, eccentricity_descriptor]
    descriptor_results = {descriptor.__name__: {'results': np.empty((10, 10)), 'class': np.arange(10) + 1} for descriptor in descriptors}

    images = {'black': [], 'contour': []}
    representative_images_number = []

    dirs = [x[0] for x in os.walk(IMAGES_DIR)]
    black_and_white_dirs = get_list_of_dirs(dirs, 'black_and_white')

    for dir_index, black_and_white_dir in enumerate(black_and_white_dirs):
        filenames = [x[2] for x in os.walk(black_and_white_dir)][0]

        for filename_iter, filename in enumerate(filenames):
            image = cv2.imread(f'{black_and_white_dir}/{filename}')
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_gray_inv = 255 - image_gray
            thresh, image_black_and_white = tresh, image_black_and_white = cv2.threshold(image_gray_inv, 0, 255, cv2.THRESH_BINARY_INV)

            for descriptor in descriptors:
                descriptor_results[descriptor.__name__]['results'][dir_index, filename_iter] = descriptor(
                    image_black_and_white)

            # cv2.imwrite(f'{contour_dirs[dir_index]}/{filename}', image_contour)

            # cv2.imshow('contours', image_contour)
            # cv2.imshow('black', image_black_and_white)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    classifiers_by_descriptor = [MLPClassifier(max_iter=1000) for _ in descriptors]
    scores = {descriptor.__name__: 0 for descriptor in descriptors}
    for clf, (descriptor_name, results) in zip(classifiers_by_descriptor, descriptor_results.items()):
        clf.fit(results['results'], results['class'])
        scores[descriptor_name] = clf.score(results['results'], results['class'])

    print(scores)

    scores = {descriptor.__name__: np.zeros(10) for descriptor in descriptors}
    for descriptor_name, results_for_descriptor in scores.items():
        for label in range(results_for_descriptor.size):
            clf = MLPClassifier(max_iter=1000)
            clf.fit(descriptor_results[descriptor_name]['results'], descriptor_results[descriptor_name]['class'])
            scores[descriptor_name][label] = clf.score(descriptor_results[descriptor_name]['results'], descriptor_results[descriptor_name]['class'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(scores).to_markdown())
