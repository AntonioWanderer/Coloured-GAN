import cv2
import os
import Config
import numpy
from tensorflow import keras


def normalize(arr: numpy.array, forward=True):
    if forward:
        arr = 2 * (arr / 255) - 1
        return arr
    else:
        arr = (arr + 1) / 2 * 255
        return arr


def generating(generator):
    results = []

    noise = numpy.random.normal(size=[10] + list(Config.imShape))

    print(noise.shape)
    prediction_array = [noise]
    for _ in range(99):
        fantasy = generator.predict(noise, verbose=False)
        fant_noise = noise - fantasy
        noise = fantasy / Config.noise_decrement + fant_noise * Config.noise_decrement
        prediction_array.append(noise)
    history_fantasy = numpy.vstack(
        [
            numpy.hstack([prediction_array[i][j] for j in range(10)]) for i in range(len(prediction_array))
        ]
    )
    print(history_fantasy.shape)
    cv2.imwrite(f"the_best/best.jpg", normalize(history_fantasy, forward=False))

    return results


if __name__ == "__main__":
    generator = keras.models.load_model("models/diffusion12560")
    results = generating(generator)
