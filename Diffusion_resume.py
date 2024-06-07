import cv2
import os
import Config
import numpy
import random
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Flatten, \
    LeakyReLU, AveragePooling2D, UpSampling2D


def get_mnist():
    (x, y), (xT, yT) = keras.datasets.mnist.load_data()
    digit = 0
    xD = []
    for i in range(x.shape[0]):
        im = x[i]
        im = cv2.resize(im, dsize=(Config.imShape[:2]))
        if True:  # y[i] == digit:
            im = numpy.repeat(numpy.expand_dims(im, axis=2), axis=2, repeats=3)
            xD.append(im)
    xD = numpy.array(xD) / Config.divisor
    # xD = numpy.repeat(numpy.expand_dims(xD, axis=3), axis=3, repeats=3)
    print(xD.shape)
    return xD


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
    data = normalize(numpy.array(data))
    print(f"Images shape: {data.shape}")
    return data


def normalize(arr: numpy.array, forward=True):
    if forward:
        arr = 2 * (arr / 255) - 1
        return arr
    else:
        arr = (arr + 1) / 2 * 255
        return arr


def train(xD, generator):
    for epoch in range(7080, 1000000):
        image_batch = numpy.array([xD[numpy.random.randint(0, xD.shape[0] - 1)] for _ in range(Config.batch_size)])
        noise = numpy.random.normal(size=image_batch.shape)
        for iteration in range(random.randint(1, Config.noise_iter)):
            noise += numpy.random.normal(size=image_batch.shape)
        mixed = image_batch + noise
        frames = [mixed[0]]
        for _ in range(Config.denoise_iter):
            generator.train_on_batch(mixed, image_batch)
            pred_im = generator.predict(mixed, verbose=False)
            pred_noise = mixed - pred_im
            mixed = pred_im / Config.noise_decrement + pred_noise * Config.noise_decrement
            frames.append(mixed[0])

        if epoch % 10 == 0:
            history_image = numpy.vstack([numpy.hstack(frames[i:i + 4]) for i in range(0, len(frames), 4)])
            print(history_image.shape)
            cv2.imwrite(f"Results/result-{epoch}-iteration.jpg", normalize(history_image, forward=False))
            score = generator.evaluate(mixed, image_batch, verbose=False)
            print(f"{epoch}: {score:.7f}")
            with open("logs.txt", "a") as f:
                f.write(str(score) + "\n")
            generator.save(f"models/diffusion{epoch}")

            prediction_array = [noise[0]]
            for _ in range(99):
                fantasy = generator.predict(noise, verbose=False)
                fant_noise = noise - fantasy
                noise = fantasy / Config.noise_decrement + fant_noise * Config.noise_decrement
                prediction_array.append(noise[0])
            history_fantasy = numpy.vstack(
                [numpy.hstack(prediction_array[i:i + 10]) for i in range(0, len(prediction_array), 10)])
            cv2.imwrite(f"Results/fantasies/fantasy-{epoch}.jpg", normalize(history_fantasy, forward=False))


if __name__ == "__main__":
    collection = get_images()
    generator = keras.models.load_model("models/diffusion7070")
    train(collection, generator)
