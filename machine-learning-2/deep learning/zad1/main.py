from typing import Tuple
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop
import cv2
from sklearn.model_selection import train_test_split

IMAGES_DIR = './images'


def adjust_data(x: np.array, y: np.array) -> Tuple:
    adjusting_x = x.reshape((x.shape[0], 28, 28, 1))
    adjusting_y = to_categorical(y)

    return adjusting_x, adjusting_y


def normalize_data(data: np.array) -> np.array:
    norm = data.astype('float32')

    return norm / 255.0


def add_noise_to_image(img: np.array, noise: float) -> np.array:
    return img + noise * np.random.randn(*img.shape)


def rotate_image_by_90_degree(img: np.array) -> np.array:
    return np.rot90(img)


def move_image_axes(img: np.array, delta: int = 2) -> np.array:
    img = np.roll(img, delta, axis=0)
    img = np.roll(img, delta, axis=1)

    return img


def zad1():
    activation_functions = ['sigmoid', 'hard_sigmoid', 'tanh', 'linear', 'relu', 'softmax']

    for activation_function in activation_functions:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation=activation_function))
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(f'\nactivation function: {activation_function}')
        model.fit(x_train, y_train)
        res = model.evaluate(x_test, y_test)
        print(f'res = {res}')

        # model.predict(image)


def zad2():
    fit_epoch_counts = [10, 100, 1000]

    for count in fit_epoch_counts:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='sigmoid'))

        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(f'\nfit epoch count: {count}')
        model.fit(x_train, y_train, epochs=count)
        res = model.evaluate(x_test, y_test)
        print(f'res = {res}')


def zad3():
    optimizers = ['adam', 'sgd', 'adadelta', 'adagrad', 'rmsprop']

    for optimizer in optimizers:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='sigmoid'))

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(f'\noptimizer: {optimizer}')
        model.fit(x_train, y_train, epochs=10)
        res = model.evaluate(x_test, y_test)
        print(f'res = {res}')


def zad4():
    learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for learning_rate in learning_rates:
        opt = RMSprop(learning_rate=learning_rate)

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='sigmoid'))

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(f'\nLearning rate: {learning_rate}')
        model.fit(x_train, y_train, epochs=10)
        res = model.evaluate(x_test, y_test)
        print(f'res = {res}')


def predict_on_image():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train)
    res = model.predict(image.reshape(1, 28, 28, 1))
    print(f'res = {res}')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    image = cv2.imread(f'{IMAGES_DIR}/sample_image.png', cv2.IMREAD_GRAYSCALE)
    image = add_noise_to_image(image, 54)
    image = rotate_image_by_90_degree(image)
    image = move_image_axes(image)

    # image.reshape(1, 28, 28, 1)
    image = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    # zad1()
    # zad2()
    # zad3()
    # zad4()
    predict_on_image()
