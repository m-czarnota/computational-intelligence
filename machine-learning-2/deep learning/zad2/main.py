from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import data

IMAGES_DIR = './images'
training_dir = 'Datasets/att_faces/Training'
testing_dir = 'Datasets/att_faces/Testing'

image_shapes = [(28, 48), (56, 96), (112, 192)]


def add_noise_to_image(img: np.array, noise: float) -> np.array:
    return img + noise * np.random.randn(*img.shape)


def create_model() -> Sequential:
    model = tf.keras.models.Sequential([
        Convolution2D(12, 3, activation='relu'),
        Convolution2D(10, 3, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=SGD(momentum=0.9))

    return model


def zad1() -> None:
    for image_shape in image_shapes:
        noise_results = []

        for noise in range(100):
            x_train, y_train = data.LoadTrainingData(training_dir, image_shape)
            x_train, y_train = x_train / 255.0, y_train / 255.0  # normalization

            for image_iter, image in enumerate(x_train):
                x_train[image_iter] = add_noise_to_image(image, noise)

            model = create_model()
            model.fit(x_train, y_train)

            x_test, y_test = data.LoadTestingData(testing_dir, image_shape)
            score = model.score(x_test, y_test)

            noise_results.append([noise, score])

        noise_results = np.array(noise_results)

        plt.figure()
        plt.plot(noise_results[:, 0], noise_results[:, 1])
        plt.title(f'Score for image size {image_shape}')
        plt.xlabel('Noise level')
        plt.ylabel('Score')
        # plt.savefig(f'{IMAGES_DIR}/zad1/image_size_{f"{image_shape}".replace("(", "").replace(")", "")}')

    plt.show()


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