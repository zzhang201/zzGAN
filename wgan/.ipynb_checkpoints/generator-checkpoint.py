import tensorflow as tf

def original_generator(zs, labels, gf_dim, num_classes):
    x = tf.keras.layers.Dense(6 * 6 * gf_dim)(zs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.reshape(x, [-1, 6, 6, gf_dim])

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(gf_dim, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(gf_dim // 2, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(1, 3, padding="same", activation='tanh')(x)

    return x
