import time
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import os

IMAGES_DIR = './images'
DATA_DIR = './data/att_faces'


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
    plot_results = []
    image_shapes = [(28, 48), (56, 96), (92, 112)]

    for image_shape in image_shapes:
        scores = []
        resized_images = resize_images(images, image_shape) if image_shape != image_shapes[-1] else images

        x_train, x_test, y_train, y_test = train_test_split(resized_images, classes, random_state=0, train_size=0.5)
        x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization

        model = create_model(tuple([*list(reversed(image_shape)), 1]))
        model.fit(x_train, y_train, epochs=100)

        for number_iter in range(500):
            random_image_number = np.random.randint(x_test.shape[0])
            x_test_random = x_test[random_image_number]
            x_test_random_resized = x_test_random.reshape(1, *list(reversed(image_shape)))

            actual_y = y_test[random_image_number]
            results = model.evaluate(x_test_random_resized, np.array([actual_y]))
            scores.append(results[1])

        plot_results.append(np.mean(scores))

    plt.figure()
    plt.bar([str(image_shape) for image_shape in image_shapes], plot_results, width=0.3, color='g')

    plt.title(f'Average score by image size')
    plt.xlabel('Random test image')
    plt.ylabel('Score')

    plt.savefig(f'{IMAGES_DIR}/zad1.png')
    # plt.show()


def zad2():
    plot_results = []
    epochs = [100, 300, 500, 1000]

    x_train, x_test, y_train, y_test = train_test_split(images, classes, random_state=0, train_size=0.5)
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization

    for epoch in epochs:
        model = create_model((112, 92, 1))

        t1 = time.time()
        model.fit(x_train, y_train, epochs=epoch)
        t2 = time.time()
        fit_time = t2 - t1

        results = model.evaluate(x_test, y_test)
        plot_results.append([fit_time, results[1]])

    plot_results = np.array(plot_results)

    plt.figure()
    plt.plot(plot_results[:, 0], plot_results[:, 1])

    plt.title('Score by fit time')
    plt.xlabel('Fit time')
    plt.ylabel('Score')

    plt.savefig(f'{IMAGES_DIR}/zad2.png')
    # plt.show()


def zad3():
    plot_results = []

    x_train, x_test, y_train, y_test = train_test_split(images, classes, random_state=0, train_size=0.5)
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization

    model = create_model((112, 92, 1))
    model.fit(x_train, y_train, epochs=100)

    for noise in range(100):
        noise_results = []

        for number_iter in range(20):
            random_image_number = np.random.randint(x_test.shape[0])
            x_test_random = x_test[random_image_number]
            x_test_random = add_noise_to_image(x_test_random, noise)

            test_loss, accuracy = model.evaluate(np.array([x_test_random]), np.array([y_test[random_image_number]]))
            noise_results.append(accuracy)

        plot_results.append([noise, np.mean(noise_results)])

    plot_results = np.array(plot_results)

    plt.figure()
    plt.plot(plot_results[:, 0], plot_results[:, 1])

    plt.title('Average score by noise')
    plt.xlabel('Noise level')
    plt.ylabel('Score')

    plt.savefig(f'{IMAGES_DIR}/zad3.png')
    # plt.show()


def zad3_inverted():
    plot_results = []

    x_train, x_test, y_train, y_test = train_test_split(images, classes, random_state=0, train_size=0.5)
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization

    for noise in range(100):
        for x_train_iter, x_train_img in enumerate(x_train):
            x_train[x_train_iter] = add_noise_to_image(x_train_img, noise)

        model = create_model((112, 92, 1))
        model.fit(x_train, y_train, epochs=100)

        noise_results = []

        for _ in range(20):
            test_loss, accuracy = model.evaluate(x_test, y_test)
            noise_results.append(accuracy)

        plot_results.append([noise, np.mean(noise_results)])

    plot_results = np.array(plot_results)

    plt.figure()
    plt.plot(plot_results[:, 0], plot_results[:, 1])

    plt.title('Score by noise')
    plt.xlabel('Noise level')
    plt.ylabel('Score')

    plt.savefig(f'{IMAGES_DIR}/zad3_inverted.png')
    # plt.show()


if __name__ == '__main__':
    images = read_images()
    classes = np.array([[number] * 10 for number in range(1, 41)]).flatten()

    zad1()
    # zad2()
    # zad3()
    # zad3_inverted()

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