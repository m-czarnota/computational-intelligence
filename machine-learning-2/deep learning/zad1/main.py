from typing import Tuple
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import cv2

IMAGES_DIR = './images'


def adjust_data(x: np.array, y: np.array) -> Tuple:
    adjusting_x = x.reshape((x.shape[0], 28, 28, 1))
    adjusting_y = to_categorical(y)

    return adjusting_x, adjusting_y


def normalize_data(data: np.array) -> np.array:
    norm = data.astype('float32')

    return norm / 255.0


def add_noise_to_image(img: np.array, noise: float) -> np.array:
    return img + noise * np.random.randn(28, 28)


def rotate_image_by_90_degree(img: np.array) -> np.array:
    return np.rot90(img)


def move_image_axes(img: np.array, delta: np.array = None) -> np.array:
    if delta is None:
        delta = np.arange(-5, 5, 0.1)

    img = np.roll(img, delta, axis=0)
    img = np.roll(img, delta, axis=1)

    return img


def zad1(data: tuple):
    x_train, y_train, x_test, y_test = data
    activation_functions = ['sigmoid', 'hard_sigmoid', 'tanh', 'linear', 'relu', 'softmax']



if __name__ == '__main__':
    mnist_data = mnist.load_data()
    adjusted_data = [adjust_data(x, y) for x, y in mnist_data]
    x_train, y_train, x_test, y_test = [normalize_data(column) for data in adjusted_data for column in data]

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', input_shape=(28 * 28, 1)))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    res = model.evaluate(x_test, y_test)
    print(f'res = {res}')

    image = cv2.imread(f'{IMAGES_DIR}/sample_image.png', cv2.IMREAD_GRAYSCALE)

