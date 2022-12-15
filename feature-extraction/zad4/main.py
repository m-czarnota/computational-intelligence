import os
import cv2

IMAGES_DIR = './images'


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color'):
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


if __name__ == '__main__':
    dirs = [x[0] for x in os.walk(IMAGES_DIR)]

    black_and_white_dirs = get_list_of_dirs(dirs, 'black_and_white')
    contour_dirs = get_list_of_dirs(dirs, 'contour')

    for dir_index, black_and_white_dir in enumerate(black_and_white_dirs):
        filenames = [x[2] for x in os.walk(black_and_white_dir)][0]

        for filename in filenames:
            image = cv2.imread(f'{black_and_white_dir}/{filename}')
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh, image_black_and_white = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY)

            image_contour = image_black_and_white.copy()
            image_contour[:] = 255

            contours, hierarchy = cv2.findContours(image_black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image=image_contour, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(f'{contour_dirs[dir_index]}/{filename}', image_contour)

            # cv2.imshow('contours', image_contour)
            # cv2.imshow('black', image_black_and_white)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
