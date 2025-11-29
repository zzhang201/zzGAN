import tensorflow as tf

class DiscriminatorModel(tf.keras.Model):
    def __init__(self, df_dim, use_sn=False, use_ln=True, name="discriminator"):
        super().__init__(name=name)
        self.use_ln = use_ln
        self.use_sn = use_sn
        c = int(df_dim)

        # 1. Define LayerNorms (Only used if use_ln=True)
        if self.use_ln:
            self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # 2. Define Layers 
        # (We use your existing SNConv2D classes from ops.py)
        # They default to SN=Enabled. We will toggle them later if use_sn=False.
        from common.model.ops import SNConv2D
        
        self.conv1 = SNConv2D(c // 4, 3, strides=(2, 2), padding="same", dtype="float32", name="conv1")
        self.act1  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.conv2 = SNConv2D(c // 2, 3, strides=(2, 2), padding="same", dtype="float32", name="conv2")
        self.act2  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.conv3 = SNConv2D(c, 3, strides=(2, 2), padding="same", dtype="float32", name="conv3")
        self.act3  = tf.keras.layers.LeakyReLU(dtype="float32")

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(dtype="float32")
        self.dense = tf.keras.layers.Dense(1, dtype="float32", name="dense")

    def build(self, input_shape):
        super().build(input_shape)
        # Logic to Disable SN if requested
        # We iterate over layers *after* they are instantiated
        if not self.use_sn:
            self._disable_sn_recursively(self)

    def _disable_sn_recursively(self, layer):
        # Helper to find SN layers and turn them off
        if hasattr(layer, 'disable_sn_now'):
            layer.disable_sn_now()
            print(f">> SN Disabled for {layer.name}")
        
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                self._disable_sn_recursively(sub)

    def call(self, inputs, training=False, return_features=False):
        x = tf.cast(inputs, tf.float32)

        # Block 1
        x = self.conv1(x, training=training)
        if self.use_ln: x = self.ln1(x) # Conditional LN
        x = self.act1(x)

        # Block 2
        x = self.conv2(x, training=training)
        if self.use_ln: x = self.ln2(x) # Conditional LN
        x = self.act2(x)

        # Block 3
        x = self.conv3(x, training=training)
        if self.use_ln: x = self.ln3(x) # Conditional LN
        x = self.act3(x)

        pooled = self.global_pool(x)
        out = self.dense(pooled)
        logits = tf.reshape(out, [-1])
        
        if return_features:
            return logits, pooled
        return logits