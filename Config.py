import tensorflow as tf
latent_dim = 100
batch_s = 64
g_resolution = 2
epochs = 200
num_img = 16
image_size=(64, 64)
seed = tf.random.normal([1, latent_dim])

