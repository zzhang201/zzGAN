import tensorflow as tf

class DiscriminatorModel(tf.keras.Model):
    def __init__(self, df_dim, name="discriminator"):
        super().__init__(name=name)
        c = int(df_dim)

        # Pin compute dtype to fp32 for stability under mixed_float16
        self.conv1 = tf.keras.layers.Conv2D(c // 4, 3, strides=2, padding="same", dtype="float32")
        self.act1  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.conv2 = tf.keras.layers.Conv2D(c // 2, 3, strides=2, padding="same", dtype="float32")
        self.act2  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.conv3 = tf.keras.layers.Conv2D(c,       3, strides=2, padding="same", dtype="float32")
        self.act3  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.global_pool   = tf.keras.layers.GlobalAveragePooling2D(dtype="float32")
        self.embedding_proj= tf.keras.layers.Dense(128, activation="relu", dtype="float32")
        self.output_layer  = tf.keras.layers.Dense(1, dtype="float32")  # logits per example

    def call(self, inputs, training=False, return_embedding=False):
        # Always compute in fp32 inside D (inputs may arrive as fp16 under AMP)
        x = tf.cast(inputs, tf.float32)          # [B, 1, L, V] (NHWC)

        x = self.conv1(x); x = self.act1(x)
        x = self.conv2(x); x = self.act2(x)
        x = self.conv3(x); x = self.act3(x)

        pooled    = self.global_pool(x)          # [B, C]
        embedding = self.embedding_proj(pooled)  # [B, 128]
        output    = self.output_layer(embedding) # [B, 1]

        if return_embedding:
            return embedding

        return tf.reshape(output, [-1])          # [B]
