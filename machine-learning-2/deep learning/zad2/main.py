from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten, MaxPool2D, Conv2D
from keras.optimizers import SGD
from keras.regularizers import l2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import os

IMAGES_DIR = './images'
DATA_DIR = './data/att_faces'

image_shapes = [(28, 48), (56, 96), (92, 112)]


def read_images() -> np.array:
    images = []

    for number in range(1, 41):
        filenames = [x[2] for x in os.walk(f'{DATA_DIR}/s{number}')][0]

        for filename in filenames:
            image = cv2.imread(f'{DATA_DIR}/s{number}/{filename}', cv2.IMREAD_GRAYSCALE)
            images.append(image)

    return np.array(images)


def resize_images(images: np.array, shape: tuple) -> np.array:
    return np.array([cv2.resize(image, shape) for image in images])


def add_noise_to_image(img: np.array, noise: float) -> np.array:
    return img + noise * np.random.randn(*img.shape)


def create_model(input_shape: tuple) -> Sequential:
    model = tf.keras.models.Sequential([
        Convolution2D(6, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Convolution2D(12, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(50, activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['accuracy'])

    return model


def zad1() -> None:
    images = read_images()
    classes = np.array([[number] * 10 for number in range(1, 41)]).flatten()

    for image_shape in image_shapes:
        noise_results = []
        resized_images = resize_images(images, image_shape)

        # resized_images = images
        # image_shape = (92, 112)

        x_train, x_test, y_train, y_test = train_test_split(resized_images, classes, random_state=0, train_size=0.5)
        x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization

        model = create_model(tuple([*list(reversed(image_shape)), 1]))
        model.fit(x_train, y_train, epochs=100)

        for noise in range(100):
            scores = []

            for _ in range(500):
                random_image_number = np.random.randint(x_test.shape[0])
                x_test_random = x_test[random_image_number]
                x_test_random = add_noise_to_image(x_test_random, noise)
                x_test_random_resized = x_test_random.reshape(1, *list(reversed(image_shape)))

                predicted_distribution = model.predict(x_test_random_resized)
                y_predicted = np.argmax(predicted_distribution)
                actual_y = y_test[random_image_number]
                scores.append(y_predicted == actual_y)

                # print(model.evaluate(x_test_random_resized, np.array([actual_y])))
                # print(f'predicted distribution:\n{predicted_distribution}')
                # print(f'draw class {y_test[random_image_number]} ?= {y_predicted} = {np.max(predicted_distribution)}, {y_predicted == actual_y}')
                # print(np.argmin(predicted_distribution))

            noise_results.append([noise, np.mean(scores)])

        noise_results = np.array(noise_results)

        plt.figure()
        plt.plot(noise_results[:, 0], noise_results[:, 1])
        plt.title(f'Score for image size {image_shape}')
        plt.xlabel('Noise level')
        plt.ylabel('Score')

        image_size_str = f"{image_shape}"\
            .replace("(", "")\
            .replace(")", "")\
            .replace(',', '')\
            .replace(' ', '_')
        plt.savefig(f'{IMAGES_DIR}/zad1_image_size_{image_size_str}.png')

    # plt.show()


if __name__ == '__main__':
    zad1()

"""
pierwsze badanie:
    małe obrazki, średnie obrazki, duże obrazki
    sprawdzamy czas uczenia i jakość dla małych, średnich, dużych
    czas uczenia będzie trochę trwać, resizing także może trochę trwać
    małe to 28x28, zachować proporcje.
    im mniejszy obrazek tym szybciej będzie się uczyć
    największy wymiar to oryginalny: 112x192
    uczymy na zaszumionych i testujemy na niezaszumionych

tam gdzie badamy rozdzielczości, to nie badamy szumów
badanie na epokach:
    wybieramy kilka wartości: 100, 200, 300
    patrzymy czym im dłużej uczymy tym lepiej jest, czy może wykres się wypłaszcza
    dla 3 wartości powinien już być mocno uśredniony wynik
    
szum:
    wpływ poziomu szumu
    mały szum, średni szum i duży szum
    cały zbiór uczący przygotowujemy z mały, średnim i dużym
    każdy obraz zaszumiony osobno
    dla każdego obrazka generować szum osobno i aplikować
    gdyby jeden szum zaaplikować do wszystkich, to słabo by to wyszło
    robimy to dla skali szarości, nie bawimy się w kolory
    
mogą być 3 konwolucyjne warstwy i 3 gęste
im więcej gęstych, tym więcej będzie trwał czas uczenia
można zastosować regularyzacje, dropout
"""