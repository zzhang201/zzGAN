import tensorflow as tf

class DiscriminatorModel(tf.keras.Model):
    def __init__(self, df_dim, name="discriminator"):
        super().__init__(name=name)
        c = int(df_dim)

        # Explicit names so ckpt keys match
        self.conv1 = tf.keras.layers.Conv2D(c // 4, 3, strides=2, padding="same",
                                            dtype="float32", name="conv1")
        self.act1  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.conv2 = tf.keras.layers.Conv2D(c // 2, 3, strides=2, padding="same",
                                            dtype="float32", name="conv2")
        self.act2  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.conv3 = tf.keras.layers.Conv2D(c,       3, strides=2, padding="same",
                                            dtype="float32", name="conv3")
        self.act3  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(dtype="float32")

        # FINAL DENSE *must* be named "dense" for ckpt:
        self.dense = tf.keras.layers.Dense(1, dtype="float32", name="dense")

    def call(self, inputs, training=False, return_features=False):
        x = tf.cast(inputs, tf.float32)          # [B, 1, L, V]
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        pooled = self.global_pool(x)             # [B, C]
        out    = self.dense(pooled)              # [B, 1]
        out    = tf.reshape(out, [-1])
        if return_features:
            return out, pooled
        return out
