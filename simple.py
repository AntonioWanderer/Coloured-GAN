import cv2
import os
import Config2
import gc
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, BatchNormalization, \
    Reshape, Flatten, LeakyReLU, AveragePooling2D, UpSampling2D, Activation, ZeroPadding2D


def get_fashion():
    (x, y), (xT, yT) = keras.datasets.fashion_mnist.load_data()
    digit = Config2.digit
    xD = []
    for i in range(x.shape[0]):
        im = x[i]
        if y[i] == digit:
            xD.append(im)
    xD = numpy.array(xD) / Config2.divisor
    xD = numpy.expand_dims(xD, axis=3)
    print(xD.shape)
    return xD


def get_mnist():
    (x, y), (xT, yT) = keras.datasets.mnist.load_data()
    digit = Config2.digit
    xD = []
    for i in range(x.shape[0]):
        im = x[i]
        im = cv2.resize(im, dsize=(64, 64))
        if y[i] == digit:
            xD.append(im)
    xD = numpy.array(xD) / Config2.divisor
    xD = numpy.expand_dims(xD, axis=3)
    print(xD.shape)
    return xD


def get_images(path=Config2.path_content):
    files = os.listdir(path)
    data = []
    for name in files:
        try:
            image = cv2.imread(filename=path + name)
            resized = cv2.resize(src=image, dsize=Config2.imShape)
            # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            data.append(resized)
            # data.append(gray)
        except:
            print("Error")
    data = numpy.array(data)
    print(data.shape)
    return data


def normalization(tensor, reverse=False):
    if reverse:
        return tensor * Config2.divisor
    else:
        return tensor / Config2.divisor


def get_model():
    generator = Sequential()
    generator.add(Dense(4 * 4 * 256, activation="relu", input_dim=Config2.noise_dim))
    generator.add(Reshape((4, 4, 256)))
    generator.add(UpSampling2D())
    generator.add(Conv2D(256, kernel_size=3, padding="same"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation("relu"))
    generator.add(UpSampling2D())
    generator.add(Conv2D(256, kernel_size=3, padding="same"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation("relu"))
    generator.add(UpSampling2D())
    generator.add(Conv2D(256, kernel_size=3, padding="same"))  #
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation("relu"))
    generator.add(UpSampling2D())
    generator.add(Conv2D(128, kernel_size=3, padding="same"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation("relu"))
    generator.add(Conv2D(1, kernel_size=3, padding="same"))
    generator.add(Activation("sigmoid"))
    generator.summary()

    generator.compile(loss="mse", optimizer="adam")

    # Building a Discriminator
    discriminator = Sequential()
    discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64, 64, 1), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation="sigmoid"))
    discriminator.summary()
    discriminator.compile(loss="binary_crossentropy", optimizer="adam")

    gan_input = Input(shape=(Config2.noise_dim,))
    g = generator(gan_input)
    gan_output = discriminator(g)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    print(generator.summary())
    print(discriminator.summary())

    return generator, discriminator, gan


def train(xD, generator, discriminator, gan):
    truth = numpy.ones(shape=(Config2.batch_size, 1))
    falth = numpy.zeros(shape=(Config2.batch_size, 1))
    for epoch in range(1000000):
        image_batch = numpy.array([xD[numpy.random.randint(0, xD.shape[0] - 1)] for _ in range(Config2.batch_size)])
        noise = numpy.random.normal(size=(Config2.batch_size, Config2.noise_dim))
        generation = generator.predict(noise, verbose=False)
        for j in range(10):
            discriminator.trainable = True
            discriminator.train_on_batch(image_batch, truth)
            discriminator.train_on_batch(generation, falth)
            discriminator.trainable = False
            gan.train_on_batch(noise, truth)
            # generator.train_on_batch(noise,image_batch)
        if epoch % 10 == 0:
            loss_T = discriminator.evaluate(image_batch, truth, verbose=False)
            loss_F = discriminator.evaluate(generation, falth, verbose=False)
            print(f"true: {loss_T}, false {loss_F}")
            for i in range(generation.shape[0]):
                cv2.imwrite(f"Results/result{epoch}-{i}.jpg", generation[i] * Config2.divisor)
            generator.save(f"models/generator{epoch}")
            discriminator.save(f"models/discriminator{epoch}")
            del loss_T
            del loss_F
        del generation
        del noise
        del image_batch
        gc.collect()
        keras.backend.clear_session()


if __name__ == "__main__":
    # collection = get_fashion()
    collection = get_mnist()
    # collection = get_images()
    generator, discriminator, gan = get_model()
    train(collection, generator, discriminator, gan)
