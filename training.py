import Models, Config, Preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    data = Preprocessing.ImportData()
    generator, discriminator, gan = Models.getModel()

    Preprocessing.Potrait_Generator(generator, filename="result.jpg")

    # history = gan.fit(data, epochs=Config.epochs,
    #                   callbacks=[ModelCheckpoint(filepath="Checkpoints/", save_best_only=True)])
    for i in range(1000):
        history = gan.fit(data, epochs=1, steps_per_epoch=10)
        Preprocessing.Potrait_Generator(generator, filename=f"result{i}.jpg")

