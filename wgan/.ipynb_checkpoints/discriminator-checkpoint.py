import tensorflow as tf

class DiscriminatorModel(tf.keras.Model):
    def __init__(self, df_dim, name="discriminator"):
        super().__init__(name=name)
        c = int(df_dim)

        # === 1. Define Layer Norms ===
        # We need one for each layer where we want to stabilize the slope.
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

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
        self.dense = tf.keras.layers.Dense(1, dtype="float32", name="dense")

    def call(self, inputs, training=False, return_features=False):
        x = tf.cast(inputs, tf.float32)

        # Block 1
        x = self.conv1(x)
        x = self.ln1(x)  # <--- Apply LN
        x = self.act1(x)

        # Block 2
        x = self.conv2(x)
        x = self.ln2(x)  # <--- Apply LN
        x = self.act2(x)

        # Block 3
        x = self.conv3(x)
        x = self.ln3(x)  # <--- Apply LN
        x = self.act3(x)

        # Output
        pooled = self.global_pool(x)
        out = self.dense(pooled)
        
        logits = tf.reshape(out, [-1])
        
        if return_features:
            return logits, pooled
        return logits