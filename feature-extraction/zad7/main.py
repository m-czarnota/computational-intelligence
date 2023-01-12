
import os
import cv2

from src.descriptor.CDP import CDP
from src.descriptor.CentroidPDH import CentroidPDH
from src.descriptor.LogPol import LogPol

IMAGES_DIR = './images'


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color'):
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


if __name__ == '__main__':
    dirs = [x[0] for x in os.walk(IMAGES_DIR)]
    contour_dirs = get_list_of_dirs(dirs, 'contour')
    bird_dirs = list(map(lambda name: name.replace('/contour', ''), contour_dirs))

    descriptors = [CDP(), LogPol(), CentroidPDH()]

    for dir_index, contour_dir in enumerate(contour_dirs):
        filenames = [x[2] for x in os.walk(contour_dir)][0]

        for filename in filenames:
            image = cv2.imread(f'{contour_dir}/{filename}')
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh, image_black_and_white = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            image_contour = image_black_and_white.copy()
            image_contour[:] = 255

            contours, hierarchy = cv2.findContours(image_black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image=image_contour, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=1)

            # cv2.imwrite(f'{contour_dirs[dir_index]}/{filename}', image_contour)
            # cv2.imshow('contours', image_contour)
            # cv2.waitKey(0)

            for descriptor in descriptors:
                descriptor.descript_image(image_contour)
                descriptor.save_image(bird_dirs[dir_index], filename)

            # cv2.imshow('contours', image_contour)
            # cv2.imshow('black', image_black_and_white)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
