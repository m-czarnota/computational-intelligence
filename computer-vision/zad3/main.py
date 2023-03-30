import os
import cv2
import matplotlib.pyplot as plt

from Vision import Vision


def get_list_of_dirs(dirs: list, sub_dir_name: str = 'color') -> list:
    list_dir = [x for x in dirs if sub_dir_name in x]
    list_dir = [x.replace("\\", "/") for x in list_dir]

    return list_dir


def create_positives_list() -> None:
    with open('pos.txt', 'w') as file:
        dirs = [x[0] for x in os.walk('./birds')]
        black_and_white_dirs = get_list_of_dirs(dirs, 'color')

        for dir_index, color_dir in enumerate(black_and_white_dirs):
            filenames = [x[2] for x in os.walk(color_dir)][0]

            for filename_iter, filename in enumerate(filenames):
                filepath = f'{color_dir}/{filename}'
                image = cv2.imread(filepath)
                file.write(f'{filepath} 1 {image.shape[0] - 1} {image.shape[1] - 1} 1 1\n')


def create_negatives_list() -> None:
    with open('neg.txt', 'w') as file:
        dogs_cats_dir = './dogs_cats'
        filenames = [x[2] for x in os.walk(dogs_cats_dir)][0]

        for filename_iter, filename in enumerate(filenames):
            filepath = f'{dogs_cats_dir}/{filename}'
            file.write(f'{filepath}\n')


def cascade_test() -> None:
    cascade_limestone = cv2.CascadeClassifier('./cascade/cascade.xml')
    vision_limestone = Vision(None)

    test_images_dir = './tests_images'
    filenames = [x[2] for x in os.walk(test_images_dir)][0]

    for filename_iter, filename in enumerate(filenames):
        filepath = f'{test_images_dir}/{filename}'
        image = cv2.imread(filepath)

        rectangles = cascade_limestone.detectMultiScale(image)
        detection_image = vision_limestone.draw_rectangles(image, rectangles)

        plt.figure()
        plt.imshow(detection_image)
        plt.imsave(f'./test_images_results/{filename}', detection_image)
        # plt.show()


if __name__ == '__main__':
    cascade_test()

    """
    1) create pos.txt
        if bounding rectangle is whole image, then subtract 1 pixel from boundary
    2) create neg.txt
    3) create sample with opencv (version 3.4)
        opencv_createsamples.exe -info pos.txt -w 48 -h 48 -num 1000 -vec pos.vec
        -w and -h are size of window to detect object
        -num is number of samples
        -vec where it should be saved
    4) you can learn cascade
        create a new folder for cascade "cascade"
        opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 48 -h 48 -numPos 100 -numNeg 200 -numStages 12
        -data is your folder for learning cascade
        -vec is positive samples
        -bg (background) negatives
        -w and -h size of window to detect object, must be the same as the values used in previous step
        -numPos is how many positive samples are use for training. Numbers for positives sample should be equals number of your positive images or few less
        -numNeg is how many negative samples are use for training. should be twice of your -numPos
        -numStages 10 is how many stages will be used for learning. more is better, but too more can over fit model
    5) when you want retrain cascade you must remove all files from your folder to learn cascade
    """
