import os
import cv2

IMAGES_DIR = './images'


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color'):
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


if __name__ == '__main__':
    dirs = [x[0] for x in os.walk(IMAGES_DIR)]

    color_dirs = get_list_of_dirs(dirs)
    gray_dirs = get_list_of_dirs(dirs, 'gray')
    black_and_white_dirs = get_list_of_dirs(dirs, 'black_and_white')

    for dir_index, color_dir in enumerate(color_dirs):
        filenames = [x[2] for x in os.walk(color_dir)][0]

        for filename in filenames:
            image = cv2.imread(f'{color_dir}/{filename}')
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray_inv = 255 - image_gray
            tresh, image_black_and_white = cv2.threshold(image_gray_inv, 0, 255, cv2.THRESH_BINARY_INV)

            cv2.imwrite(f'{gray_dirs[dir_index]}/{filename}', image_gray)
            cv2.imwrite(f'{black_and_white_dirs[dir_index]}/{filename}', image_black_and_white)

            # cv2.imshow('gray', image_gray_inv)
            # cv2.imshow('black', image_black_and_white)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
