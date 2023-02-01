from typing import Tuple
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.optimizers.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow import Tensor
from tensorflow.python.checkpoint.checkpoint import Checkpoint
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import time
import cv2
import pickle


# BUFFER_SIZE = 60000
BUFFER_SIZE = 50000
# BATCH_SIZE = 256
BATCH_SIZE = 500

IMAGES_DIR = './images'
ANIMATIONS_DIR = './animations'
CHECKPOINT_DIR = './training_checkpoints'
EPOCH_IMAGES_DIR = f'{IMAGES_DIR}/epoch'

labels: dict = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


def make_generator_model(resolution: int, noise_dim: int) -> Sequential:
    r2 = int(resolution / 2)
    r4 = int(resolution / 4)

    model = Sequential()
    model.add(Dense(r4 * r4 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((r4, r4, 256)))
    assert model.output_shape == (None, r4, r4, 256)  # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, r4, r4, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, r2, r2, 64)

    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, resolution, resolution, 3)

    return model


def make_discriminator_model(resolution: int) -> Sequential:
    """
    Discriminator returns positives values for real images and negatives values for fake images,
    """
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(resolution, resolution, 3)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def discriminator_loss(real_output: Tensor, fake_output: Tensor) -> Tensor:
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def generator_loss(fake_output: Tensor) -> Tensor:
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(images: np.array, generator: Sequential, discriminator: Sequential, generator_optimizer: OptimizerV2, discriminator_optimizer: OptimizerV2, noise_dim: int) -> Tuple:
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_save_images(model: Sequential, epoch: int, test_input: np.array) -> None:
    predictions = model(test_input, training=False)
    predictions = predictions * 127.5 + 127.5
    predictions = np.array(predictions).astype("uint8")

    fig = plt.figure(figsize=(4, 4))
    k = int(np.sqrt(num_examples_to_generate))

    for i in range(predictions.shape[0]):
        plt.subplot(k, k, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig(f'{EPOCH_IMAGES_DIR}/image_at_epoch_{epoch}.png')
    plt.close(fig)
    # plt.show() if epoch % 10 == 0 or epoch == 1 else plt.close(fig)


def train(dataset: DatasetV1Adapter, generator: Sequential, discriminator: Sequential, generator_optimizer: OptimizerV2, discriminator_optimizer: OptimizerV2, epochs: int, noise_dim: int, seed: np.array, checkpoint: Checkpoint) -> np.array:
    losses = np.zeros((epochs * len(dataset), 2))  # gen, disc
    idx = 0

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            losses[idx] = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim)
            idx += 1

        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    generate_and_save_images(generator, epochs, seed)

    return losses


def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        unpicked = pickle.load(fo, encoding='bytes')

    return unpicked


def read_images_train(class_label: str) -> Tuple:
    label = labels[class_label]
    train_images, train_labels = [], []

    for i in np.arange(1, 6):
        data = unpickle(f"cifar/data_batch_{i}")

        data_images = data[b'data']
        data_labels = np.array(data[b'labels'])

        train_images.extend(data_images[data_labels == label, :])
        train_labels.extend(list(data_labels[data_labels == label]))

    return np.array(train_images), np.array(train_labels)


def reduce_resolution(images: np.array, resolution: int) -> np.array:
    tr = np.zeros((images.shape[0], resolution, resolution, 3)).astype("uint8")

    for i, img in enumerate(images):
        tr[i] = cv2.resize(img, (resolution, resolution))

    return tr


def generate_gif(anim_filename: str) -> None:
    with imageio.get_writer(anim_filename, mode='I') as writer:
        filenames = glob.glob(f'{EPOCH_IMAGES_DIR}/image*.png')
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)


def read_train_data(resolution: int = 32) -> Tuple:
    images, image_labels = read_images_train(label_class)
    images = np.transpose(images.reshape(images.shape[0], 32, 32, 3, order='F'), axes=[0, 2, 1, 3])
    images = reduce_resolution(images, resolution) if resolution != 32 else images  # resolution reduce
    images = images.astype('float32')
    images = (images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    dataset = Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return dataset, image_labels


def plot_losses_functions(epochs: int, fit_time: float, losses: np.array, noise_dim: int, file_name: str) -> None:
    fig, ax = plt.subplots()
    ax.set_title(f'Gen and disc losses for {epochs} epochs with initializing vector size {noise_dim}')

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0, -0.07, f'Fitting time: {fit_time:.2f}s', transform=trans, va='top', ha='center', color='r')

    xd = np.arange(0, losses.shape[0])
    ax.plot(xd, losses[:, 0], label="gen_loss")
    ax.plot(xd, losses[:, 1], label="disc_loss")

    ax.legend()
    plt.savefig(f'{IMAGES_DIR}/{file_name}')
    plt.close()


def zad1_initializing_vector() -> None:
    noise_dims = [50, 100, 200]
    fit_times = np.zeros(len(noise_dims))

    epochs = 100
    resolution = 32  # Resolution min=4, default=32, only multiples of 4

    train_dataset, train_labels = read_train_data(resolution)

    for noise_dim in noise_dims:
        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        generator = make_generator_model(resolution=resolution, noise_dim=noise_dim)
        discriminator = make_discriminator_model(resolution=resolution)

        generator_optimizer = Adam(1e-4)
        discriminator_optimizer = Adam(1e-4)
        checkpoint = Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator, discriminator=discriminator)

        t1 = time.time()
        losses = train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs,
                       noise_dim, seed, checkpoint)
        t2 = time.time()
        fit_time = t2 - t1

        plot_losses_functions(epochs, fit_time, losses, noise_dim, f'zad1_loss_graph_initializing_dim_{noise_dim}.png')
        generate_gif(f'{ANIMATIONS_DIR}/zad1_dcgan_initializing_dim_{noise_dim}.gif')

    fig = plt.figure()
    plt.bar(list(map(lambda x: str(x), noise_dims)), fit_times)
    plt.title('Fitting time by initializing vector size')
    plt.savefig(f'{IMAGES_DIR}/zad1_initializing_vector_times.png')
    plt.close(fig)


def zad2() -> None:
    epochs_counts = [100, 300, 500]
    fit_times = np.zeros(len(epochs_counts))

    resolution = 32  # Resolution min=4, default=32, only multiples of 4
    noise_dim = 100  # Dim of generator's initialization vector
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train_dataset, train_labels = read_train_data(resolution)

    for epochs_iter, epochs in enumerate(epochs_counts):
        generator = make_generator_model(resolution=resolution, noise_dim=noise_dim)
        discriminator = make_discriminator_model(resolution=resolution)

        generator_optimizer = Adam(1e-4)
        discriminator_optimizer = Adam(1e-4)
        checkpoint = Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator, discriminator=discriminator)

        t1 = time.time()
        losses = train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs, noise_dim, seed, checkpoint)
        t2 = time.time()
        fit_time = t2 - t1

        fit_times[epochs_iter] = fit_time
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

        plot_losses_functions(epochs, fit_time, losses, noise_dim, f'zad2_loss_graph_epochs{epochs}.png')
        generate_gif(f'{ANIMATIONS_DIR}/zad2_dcgan_epochs_{epochs}.gif')

    fig = plt.figure()
    plt.bar(list(map(lambda x: str(x), epochs_counts)), fit_times)
    plt.title('Fitting time by epochs')
    plt.savefig(f'{IMAGES_DIR}/zad2_fit_times.png')
    plt.close(fig)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    num_examples_to_generate = 16  # Count of examples displaying for each epoch, only powers of 4
    label_class = "dog"  # Object's class name to generate

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    cross_entropy = BinaryCrossentropy(from_logits=True)

    zad1_initializing_vector()
    zad2()
