# import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(2)
import Models, Config, Preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint
from time import sleep


if __name__ == "__main__":
    data = Preprocessing.ImportData()
    generator, discriminator, gan = Models.getModel()

    Preprocessing.Potrait_Generator(generator, filename="result.jpg")

    # history = gan.fit(data, epochs=Config.epochs,
    #                   callbacks=[ModelCheckpoint(filepath="Checkpoints/", save_best_only=True)])
    for i in range(1000):
        sleep(10)
        history = gan.fit(data, epochs=1, steps_per_epoch=5)
        Preprocessing.Potrait_Generator(generator, filename=f"result{i}.jpg")
        generator.save(f"Checkpoints/generator{i}")
        discriminator.save(f"Checkpoints/discriminator{i}")