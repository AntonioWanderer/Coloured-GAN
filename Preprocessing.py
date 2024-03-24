import Config
import tensorflow as tf
import cv2

def ImportData(data_path="Content/Portraits/"):
    data = tf.keras.preprocessing.image_dataset_from_directory(data_path, label_mode=None, image_size=Config.image_size,
                                                               batch_size=Config.batch_s)
    return data

def Potrait_Generator(generator, filename="result.jpg"):
    seed = tf.random.normal([Config.num_img, Config.latent_dim])
    generated_image = generator.predict(seed)
    generated_image *= 255
    for i in range(Config.num_img):
        cv2.imwrite(f"Results/{filename}", generated_image[i])