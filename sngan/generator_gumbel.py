# gan/sngan/generator_gumbel.py
import tensorflow as tf
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from common.model.ops import (
    SNConv2D, SNConv2DTranspose, SNLinear, FinalBN, BatchNorm, leaky_relu, 
    ResBlock, RefineBlock # Ensure RefineBlock is imported!
)

NUM_AMINO_ACIDS = 21 

class GumbelGenerator(tf.keras.Model):
    def __init__(self, config, shape, num_classes=None, name="generator"):
        super().__init__(name=name)
        self.config = config
        self.shape = shape
        self.dim = config.gf_dim
        self.batch_size = shape[0]
        self.height = shape[1] 
        self.width = shape[2]  
        self.channels = shape[3]
        
        # We hardcode 5 layers based on 160 width logic (160 / 2^5 = 5)
        self.num_layers = 5
        self.starting_dim = int(self.dim * (2 ** self.num_layers))
        self.initial_width = self.width // (2 ** self.num_layers)
        self.initial_shape = (1, self.initial_width, self.starting_dim)

        # ---- Bookends ----
        self.noise_fc = SNLinear(
            units=self.initial_shape[0] * self.initial_shape[1] * self.initial_shape[2],
            name="noise_fc", dtype="float32"
        )

        # =========================================================
        # THE SAVING FIX + THE ARCHITECTURE FIX (MERGED)
        # =========================================================
        # We do NOT use self.res_blocks = [] lists.
        # We loop and use setattr to force TensorFlow to track variables.
        
        for i in range(self.num_layers):
            # 1. Calculate Dim
            doubling_point = self.num_layers - 2
            layer_dim = self.starting_dim if i < doubling_point else self.starting_dim // 2
            
            # 2. Instantiate Layers
            #    (Note: Stride (1,2) hardcoded for upsampling logic)
            res = ResBlock(hidden_dim=layer_dim, stride=(1,2), name=f"res_block_{i}")
            
            #    (Note: Dilation increases: 1, 2, 4, 8, 16)
            ref = RefineBlock(filters=layer_dim, dilation=2**i, name=f"refine_{i}")
            
            # 3. FORCE TRACKING (The fix for your Paranoia Check)
            setattr(self, f"res_block_{i}", res)
            setattr(self, f"refine_{i}", ref)
        # =========================================================

        self.final_bn  = FinalBN(name="final_bn")
        self.last_conv = SNConv2D(filters=NUM_AMINO_ACIDS, kernel_size=(1, 1),
                                  name="last_conv", dtype="float32")

        # ---- Attention (Optional/Configurable) ----
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64,
                                                       name="attn", dtype="float32")
        self.attn_ln = tf.keras.layers.LayerNormalization(name="attn_ln", dtype="float32")
        self.attn_block_index = config.attn_pos 

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
        """Injects linear coordinate ramp (-1 to 1)"""
        shape = tf.shape(x)
        width = shape[2]
        pos = tf.linspace(-1.0, 1.0, width)
        pos = tf.reshape(pos, [1, 1, width, 1])
        pos = tf.cast(pos, dtype=x.dtype)
        pos = tf.tile(pos, [shape[0], 1, 1, 1])
        return tf.concat([x, pos], axis=-1)

    def call(self, z, training=False, return_hard=False, return_attention=False, return_embedding=False):
        z = tf.cast(z, tf.float32)

        # 1. Start
        h = self.noise_fc(z)
        h = tf.reshape(h, (tf.shape(z)[0], *self.initial_shape))
        h = self._add_gps(h) # Injection 1

        self.last_attn_scores = None

        # 2. Loop through named layers
        for i in range(self.num_layers):
            # RETRIEVE LAYERS BY NAME (Matching the __init__ fix)
            res_block = getattr(self, f"res_block_{i}")
            refine_block = getattr(self, f"refine_{i}")
            
            # A. Upsample
            h = res_block(h, training=training)
            
            # B. Refine (Dilated)
            h = refine_block(h, training=training)
            
            # C. GPS Refresh
            h = self._add_gps(h)

            # D. Attention
            if i == self.attn_block_index:
                h_shape = tf.shape(h)
                h_flat = tf.reshape(h, [h_shape[0], h_shape[2], h_shape[3]])
                h_attn, scores = self.attn(h_flat, h_flat, h_flat,
                                             return_attention_scores=True, training=training)
                h = self.attn_ln(h_flat + h_attn, training=training)
                if return_attention:
                    self.last_attn_scores = scores
                h = tf.reshape(h, [h_shape[0], 1, h_shape[2], h_shape[3]])

        # 3. Final
        h = self.final_bn(h, training=training)
        h = leaky_relu(h, alpha=0.2)

        if return_embedding:
            return tf.reduce_mean(h, axis=[1, 2])

        logits = self.last_conv(h)
        
        if return_hard:
            return tf.one_hot(tf.argmax(logits, axis=-1), depth=21)
        else:
            temperature = tf.cast(self.get_temperature(training=training), tf.float32)
            dist = RelaxedOneHotCategorical(temperature=temperature, logits=logits)
            return dist.sample()