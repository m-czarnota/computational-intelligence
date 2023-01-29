import time
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.datasets import cifar10
from keras.optimizers import Adam

IMAGES_DIR = './images'


def create_discriminator(in_shape: tuple = (32, 32, 3)) -> Sequential:
    model = Sequential([
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),

        Flatten(),
        Dropout(0.4),
        Dense(1, activation='sigmoid'),
    ])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def create_generator(latent_dim: int) -> Sequential:
    n_nodes = 128 * 8 * 8
    model = Sequential([
        Dense(n_nodes, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Reshape((8, 8, 128)),

        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(3, (8, 8), activation='tanh', padding='same'),
    ])

    return model


def create_gan(generator: Sequential, discriminator: Sequential) -> Sequential:
    model = Sequential([
        generator,
        discriminator,
    ])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


def load_real_samples() -> np.array:
    (train_x, _), (_, _) = cifar10.load_data()
    x = train_x.astype('float32')
    x = (x - 127.5) / 127.5

    return x


def generate_real_samples(dataset: np.array, n_samples: int) -> Tuple:
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = np.ones((n_samples, 1))

    return x, y


def generate_latent_points(latent_dim: int, n_samples: int) -> np.array:
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input


def generate_fake_samples(generator: Sequential, latent_dim: int, n_samples: int) -> Tuple:
    x_input = generate_latent_points(latent_dim, n_samples)
    x = generator.predict(x_input)
    y = np.zeros((n_samples, 1))

    return x, y


def train(g_model: Sequential, d_model: Sequential, gan_model: Sequential, dataset: np.array, latent_dim: int, epochs: int = 100, n_batch: int = 128) -> Tuple:
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    generator_losses = []
    discriminator_losses_real = []
    discriminator_losses_fake = []

    for epoch in range(epochs):
        for batch_size in range(bat_per_epo):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _ = d_model.train_on_batch(x_real, y_real)

            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)

            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

            x_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))

            g_loss = gan_model.train_on_batch(x_gan, y_gan)

            generator_losses.append(g_loss)
            discriminator_losses_real.append(d_loss_real)
            discriminator_losses_fake.append(d_loss_fake)

            print(f'Epoch>{epoch + 1}, Batch {batch_size}/{bat_per_epo}, d1={d_loss_real:.3f}, d2={d_loss_fake:.3f} g={g_loss:.3f}')

    g_model.save('cifar_generator_2epochs.h5')

    return generator_losses, discriminator_losses_real, discriminator_losses_fake


def zad2() -> None:
    epochs = [100, 200, 300]
    plot_results = []

    latent_dim = 100
    real_samples = load_real_samples()

    discriminator = create_discriminator()
    generator = create_generator(latent_dim)
    gan = create_gan(generator, discriminator)

    for epoch in epochs:
        t1 = time.time()
        g_losses, d_losses_real, d_losses_fake = train(generator, discriminator, gan, real_samples, latent_dim, epochs=epoch, n_batch=128)
        t2 = time.time()
        fit_time = t2 - t1

        plt.figure()
        plt.title('Loss by fit time')
        plt.xlabel('Time (s)')
        plt.ylabel('Loss')

        time_vector = np.arange(fit_time)
        plt.plot(time_vector, g_losses, label='Generator')
        plt.plot(time_vector, d_losses_real, label='Discriminator real')
        plt.plot(time_vector, d_losses_fake, label='Discriminator fake')

        plt.legend()
        plt.savefig(f'{IMAGES_DIR}/zad1_epochs_{epoch}.png')
        # plt.show()

        plot_results.append([fit_time, np.mean(g_losses), np.mean(d_losses_real), np.mean(d_losses_fake)])

    plot_results = np.array(plot_results)

    plt.figure()
    plt.title('Average loss by fit time')
    plt.xlabel('Time (s)')
    plt.ylabel('Average loss')

    plt.plot(plot_results[:, 0], plot_results[:, 1], label='Generator')
    plt.plot(plot_results[:, 0], plot_results[:, 2], label='Discriminator real')
    plt.plot(plot_results[:, 0], plot_results[:, 3], label='Discriminator fake')

    plt.legend()
    plt.savefig(f'{IMAGES_DIR}/zad1.png')
    # plt.show()


if __name__ == '__main__':
    zad2()
