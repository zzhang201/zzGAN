# gan/sngan/generator_gumbel.py
import tensorflow as tf
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from common.model.ops import (
    SNConv2D, SNConv2DTranspose, SNLinear, FinalBN, BatchNorm, leaky_relu, ResBlock
)

NUM_AMINO_ACIDS = 21 

class GumbelGenerator(tf.keras.Model):
    def __init__(self, config, shape, num_classes=None, name="generator"):
        super().__init__(name=name)
        self.config = config
        self.shape = shape
        self.num_classes = num_classes
        self.dim = config.gf_dim
        self.batch_size = shape[0]
        self.height = shape[1]   # expect 1
        self.width = shape[2]    # expect 160
        self.channels = shape[3]
        self.strides = self.get_strides()
        self.number_of_layers = len(self.strides)
        self.starting_dim = int(self.dim * (2 ** self.number_of_layers))
        self.initial_width = self.width // (2 ** self.number_of_layers)
        self.initial_shape = (1, self.initial_width, self.starting_dim)

        # ---- Bookends ----
        self.noise_fc = SNLinear(
            units=self.initial_shape[0] * self.initial_shape[1] * self.initial_shape[2],
            name="noise_fc", dtype="float32"
        )

        self.res_blocks = [
            ResBlock(
                hidden_dim=self._get_hidden_dim_for_layer(i), 
                stride=stride, 
                name=f"res_block_{i}"
            ) for i, stride in enumerate(self.strides)
        ]

        self.final_bn  = FinalBN(name="final_bn")
        self.last_conv = SNConv2D(filters=NUM_AMINO_ACIDS, kernel_size=(1, 1),
                                  name="last_conv", dtype="float32")

        # ---- Attention & LayerNorm ----
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64,
                                                       name="attn", dtype="float32")
        self.attn_ln = tf.keras.layers.LayerNormalization(name="attn_ln", dtype="float32")

        # Control
        self.attn_block_index = config.attn_pos 
        self.last_attn_scores = None
        self.last_logits = None
        self.last_probs = None

    def _get_hidden_dim_for_layer(self, layer_id):
        doubling_point = self.number_of_layers - 2
        if layer_id >= doubling_point:
            return self.starting_dim * 2
        else:
            return self.starting_dim
            
    def build(self, input_shape):
        # ... (Same as your original code) ...
        h = tf.TensorShape([self.batch_size, *self.initial_shape]) 
        for i, block in enumerate(self.res_blocks):
            h = block.compute_output_shape(h)
            if i == self.attn_block_index:
                b, _, w, c = h
                attn_input_shape = tf.TensorShape([b, w, c])
                if not self.attn.built:
                    self.attn.build(query_shape=attn_input_shape,
                                    value_shape=attn_input_shape,
                                    key_shape=attn_input_shape)
                break 
        super().build(input_shape)
    
    def get_strides(self):
        return [(1, 2)] * 5

    def get_temperature(self, training=True):
        if not training:
            return tf.constant(0.5, dtype=tf.float32)
    
        start_temp  = tf.constant(1.0, dtype=tf.float32)
        end_temp    = tf.constant(0.5, dtype=tf.float32)
        target_steps = tf.constant(100_000.0, dtype=tf.float32) 
    
        decay = tf.math.log(start_temp / end_temp) / target_steps
        step = tf.cast(getattr(self, "global_step", 0), tf.float32)
    
        tau = start_temp * tf.exp(-decay * step)
        return tf.maximum(end_temp, tau)


    def _add_gps(self, x):
        """
        Helper: Injects a linear coordinate ramp (-1 to 1) 
        matching the current shape of x.
        """
        # x shape: [Batch, 1, Width, Channels]
        shape = tf.shape(x)
        batch_size = shape[0]
        width = shape[2]
        
        # 1. Create Linear Ramp
        pos = tf.linspace(-1.0, 1.0, width) # [Width]
        pos = tf.reshape(pos, [1, 1, width, 1]) # [1, 1, W, 1]
        pos = tf.cast(pos, dtype=x.dtype)
        
        # 2. Tile to batch size
        pos = tf.tile(pos, [batch_size, 1, 1, 1])
        
        # 3. Concatenate
        return tf.concat([x, pos], axis=-1)
        
    def call(self, z, training=False, return_hard=False, return_attention=False, return_embedding=False):
        # Keep compute in fp32 inside; SN vars are fp32
        z = tf.cast(z, tf.float32)

        # FC to initial 1 x (width//32) x channels
        h = self.noise_fc(z)
        h = tf.reshape(h, (tf.shape(z)[0], *self.initial_shape))  # [B, 1, W0, C]

        # =========================================================
        # === GPS / POSITIONAL ENCODING (The "Coordinate Grid") ===
        # =========================================================
        # === INJECTION 1: The Coarse GPS (Length 5) ===
        h = self._add_gps(h) 

        self.last_attn_scores = None
        
        # 2. Loop through ResBlocks
        for i, block in enumerate(self.res_blocks):
            
            # === INJECTION 2...N: The Fine GPS ===
            h = block(h, training=training) # Upsamples (e.g., 5 -> 10)
            h = self._add_gps(h) 

            # Attention Logic
            if i == self.attn_block_index:
                h_shape = tf.shape(h)
                h_flat = tf.reshape(h, [h_shape[0], h_shape[2], h_shape[3]]) 
                h_attn, scores = self.attn(h_flat, h_flat, h_flat,
                                             return_attention_scores=True, training=training)
                h = self.attn_ln(h_flat + h_attn, training=training)
                if return_attention:
                    self.last_attn_scores = scores
                h = tf.reshape(h, [h_shape[0], 1, h_shape[2], h_shape[3]])

        # Final processing
        h = self.final_bn(h, training=training)
        h = leaky_relu(h, alpha=0.2)

        if return_embedding:
            return tf.reduce_mean(h, axis=[1, 2])

        # Project to vocab logits
        logits = self.last_conv(h)    # [B, 1, W, V]
        logits = tf.cast(logits, tf.float32)
        self.last_logits = logits
        self.last_probs = tf.nn.softmax(logits, axis=-1)

        # Gumbel-Softmax (relaxed) or hard argmax
        if return_hard:
            hard = tf.one_hot(tf.argmax(logits, axis=-1), depth=logits.shape[-1])
            return hard  # [B, 1, W, V]
        else:
            temperature = tf.cast(self.get_temperature(training=training), tf.float32)
            dist = RelaxedOneHotCategorical(temperature=temperature, logits=logits)
            soft = dist.sample()
            return soft  # [B, 1, W, V]