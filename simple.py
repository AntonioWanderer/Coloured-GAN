import cv2
import os
import Config
import numpy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Flatten, \
    LeakyReLU, AveragePooling2D, UpSampling2D


def get_images(path=Config.path_content):
    files = os.listdir(path)
    data = []
    for name in files:
        try:
            image = cv2.imread(filename=path + name)
            resized = cv2.resize(src=image, dsize=Config.imShape[:2])
            data.append(resized)
        except:
            print("Error")
    data = numpy.array(data) / Config.divisor
    print(f"Images shape: {data.shape}")
    return data


def get_model():
    generator = Sequential()
    for i in range(Config.Unet_blocks):
        if i == 0:
            generator.add(
                Conv2D(filters=3 * Config.pool_kernel ** (2 * (i + 1)), kernel_size=Config.conv_kernel, padding="same",
                       activation=Config.in_activation, input_shape=Config.imShape))
        else:
            generator.add(
                Conv2D(filters=3 * Config.pool_kernel ** (2 * (i + 1)), kernel_size=Config.conv_kernel, padding="same",
                       activation=Config.in_activation))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(AveragePooling2D(pool_size=(Config.pool_kernel, Config.pool_kernel)))
    for i in range(Config.Unet_blocks - 1, -1, -1):
        generator.add(Conv2DTranspose(filters=3 * Config.pool_kernel ** (2 * (i + 1)), kernel_size=Config.conv_kernel,
                                      padding="same", activation=Config.in_activation))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(UpSampling2D(size=(Config.pool_kernel, Config.pool_kernel)))
    generator.add(Conv2DTranspose(filters=3, kernel_size=Config.conv_kernel, padding="same", activation="sigmoid"))

    generator.summary()

    generator.compile(loss="mse", optimizer="adam")

    # Building a Discriminator
    discriminator = Sequential()
    for i in range(Config.discriminator_blocks):
        if i == 0:
            discriminator.add(
                Conv2D(filters=3 * Config.pool_kernel ** (2 * (i + 1)), kernel_size=Config.conv_kernel, padding="same",
                       activation=Config.in_activation, input_shape=Config.imShape))
        else:
            discriminator.add(
                Conv2D(filters=3 * Config.pool_kernel ** (2 * (i + 1)), kernel_size=Config.conv_kernel, padding="same",
                       activation=Config.in_activation))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(AveragePooling2D(pool_size=(Config.pool_kernel, Config.pool_kernel)))

    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation="sigmoid"))
    discriminator.summary()
    discriminator.compile(loss="binary_crossentropy", optimizer="adam")

    gan_input = Input(shape=Config.imShape)
    g = generator(gan_input)
    gan_output = discriminator(g)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    print(generator.summary())
    print(discriminator.summary())

    return generator, discriminator, gan


def normalize(arr: numpy.array):
    arr -= arr.min()
    arr /= arr.max()
    return arr


def train(xD, generator, discriminator, gan):
    truth = numpy.ones(shape=(Config.batch_size, 1))
    falth = numpy.zeros(shape=(Config.batch_size, 1))
    for epoch in range(1000000):
        image_batch = numpy.array([xD[numpy.random.randint(0, xD.shape[0] - 1)] for _ in range(Config.batch_size)])
        noise = numpy.random.normal(size=image_batch.shape)
        mixed = noise + image_batch
        generation = generator.predict(noise, verbose=False)

        discriminator.trainable = False
        for _ in range(10):
            generator.train_on_batch(mixed, image_batch)
        for _ in range(10):
            gan.train_on_batch(noise, truth)
        discriminator.trainable = True
        for _ in range(10):
            discriminator.train_on_batch(image_batch, truth)
            discriminator.train_on_batch(generation, falth)

        if epoch % 10 == 0:
            loss_T = discriminator.evaluate(image_batch, truth, verbose=False)
            loss_F = discriminator.evaluate(generation, falth, verbose=False)
            print(f"true: {loss_T:.7f}, false {loss_F:.7f}")
            cv2.imwrite(f"Results/result{epoch}-{0}.jpg", generation[0] * Config.divisor)
            generator.save(f"models/generator{epoch}")
            discriminator.save(f"models/discriminator{epoch}")


if __name__ == "__main__":
    collection = get_images()
    generator, discriminator, gan = get_model()
    train(collection, generator, discriminator, gan)
