from typing import Tuple
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, MaxPooling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.optimizers import Adam
from tensorflow.python.checkpoint.checkpoint import Checkpoint
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import cv2
import pickle


# BUFFER_SIZE = 60000
BUFFER_SIZE = 50000
# BATCH_SIZE = 256
BATCH_SIZE = 500

IMAGES_DIR = './images'
CHECKPOINT_DIR = './training_checkpoints'
EPOCH_IMAGES_DIR = f'{IMAGES_DIR}/epoch'

labels = {
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
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((r4, r4, 256)))
    assert model.output_shape == (None, r4, r4, 256)  # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, r4, r4, 128)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, r2, r2, 64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
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


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images: np.array) -> Tuple:
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
    # plt.show() if epoch % 10 == 0 or epoch == 1 else plt.close(fig)


def train(dataset: DatasetV1Adapter, epochs: int) -> None:
    losses = np.zeros((epochs * len(dataset), 2))  # gen, disc
    idx = 0

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            losses[idx] = train_step(image_batch)
            idx += 1

        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    plt.figure()
    xd = np.arange(0, idx)

    plt.plot(xd, losses[:, 0], label="gen_loss")
    plt.plot(xd, losses[:, 1], label="disc_loss")

    plt.legend()
    plt.savefig('loss_graph.png')
    plt.show()

    generate_and_save_images(generator, epochs, seed)


def display_image(epoch_no):
    return PIL.Image.open('epoch_images/image_at_epoch_{:04d}.png'.format(epoch_no))


def unpickle(file):
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


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    EPOCHS = 200  # Count of epochs
    noise_dim = 100  # Dim of generator's initialization vector
    num_examples_to_generate = 16  # Count of examples displaying for each epoch, only powers of 4
    resolution = 32  # Resolution min=4, default=32, only multiples of 4
    label_class = "dog"  # Object's class name to generate

    train_images, train_labels = read_images_train(label_class)
    train_images = np.transpose(train_images.reshape(train_images.shape[0], 32, 32, 3, order='F'), axes=[0, 2, 1, 3])
    train_images = reduce_resolution(train_images, resolution) if resolution != 32 else train_images  # resolution reduce
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5   # Normalize the images to [-1, 1]

    train_dataset = Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    generator = make_generator_model(resolution=resolution, noise_dim=noise_dim)
    discriminator = make_discriminator_model(resolution=resolution)

    cross_entropy = BinaryCrossentropy(from_logits=True)
    generator_optimizer = Adam(1e-4)
    discriminator_optimizer = Adam(1e-4)

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                            generator=generator, discriminator=discriminator)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    train(train_dataset, EPOCHS)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

    anim_file = 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f'{EPOCH_IMAGES_DIR}/image*.png')
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)
