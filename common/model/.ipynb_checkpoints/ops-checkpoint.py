# common/model/ops.py
import tensorflow as tf
from tensorflow.keras import initializers

# ---------- Small utils ----------
def leaky_relu(x, alpha=0.2, name=None):
    return tf.nn.leaky_relu(x, alpha=alpha, name=name)

def dense(x, units, name=None):
    return tf.keras.layers.Dense(units, name=name)(x)

def flatten(x):
    return tf.reshape(x, [tf.shape(x)[0], -1])

def reshape(x, shape):
    return tf.reshape(x, shape)

# ---------- BatchNorm (subclass, no extra wrapper names) ----------
class BatchNorm(tf.keras.layers.BatchNormalization):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x, training=True):
        return super().call(x, training=training)

class FinalBN(tf.keras.layers.Layer):
    """Places BN stats at generator/final_bn/bn/{moving_mean,moving_variance}."""
    def __init__(self, name="final_bn"):
        super().__init__(name=name)
        self.bn = BatchNorm(name="bn")

    def call(self, x, training=False):
        return self.bn(x, training=training)

# ---------- Spectral Normalized Linear ----------
class SNLinear(tf.keras.layers.Layer):
    """
    Linear layer with spectral normalization.
    Variable names match legacy ckpt:
      w: (in_dim, units)
      b: (units,)    [optional]
      u: (1, units)  [SN power-iter vector]
    """
    def __init__(self, units, use_bias=True, power_iters=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        self.power_iters = int(power_iters)
        # Runtime toggle for SN (False = SN enabled, True = disabled)
        self.disable_sn = tf.Variable(False, trainable=False, dtype=tf.bool, name="disable_sn")

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        var_dtype = tf.float32  # keep vars in fp32 for stability

        self.w = self.add_weight(
            name="w", shape=(in_dim, self.units),
            initializer="glorot_uniform", dtype=var_dtype, trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b", shape=(self.units,),
                initializer="zeros", dtype=var_dtype, trainable=True
            )
        else:
            self.b = None

        self.u = self.add_weight(
            name="u", shape=(1, self.units),
            initializer=initializers.RandomNormal(mean=0.0, stddev=1.0),
            dtype=var_dtype, trainable=False
        )
        super().build(input_shape)

    def _spectral_norm(self, w):
        # w: [M, N]  ->  SN over N
        w_mat = tf.reshape(w, [-1, self.units])
        u = tf.stop_gradient(self.u)
        eps = 1e-12
        for _ in range(self.power_iters):
            v = tf.nn.l2_normalize(tf.matmul(u, w_mat, transpose_b=True), epsilon=eps)  # [1, M]
            u = tf.nn.l2_normalize(tf.matmul(v, w_mat), epsilon=eps)                    # [1, N]
        sigma = tf.matmul(tf.matmul(v, w_mat), u, transpose_b=True)  # [1,1]
        sigma = tf.stop_gradient(sigma)
        w_sn = tf.reshape(w_mat / (sigma + eps), tf.shape(w))
        # Preserve gradients w.r.t. original w
        return w_sn + 0.0 * (w - tf.stop_gradient(w))

    def call(self, x, training=False):
        x = tf.cast(x, self.w.dtype)
        w_use = tf.cond(self.disable_sn, lambda: self.w, lambda: self._spectral_norm(self.w))
        y = tf.matmul(x, w_use)
        if self.b is not None:
            y = y + self.b
        return y

    # Convenience toggles
    def enable_sn(self, enabled=True):
        self.disable_sn.assign(not bool(enabled))

    def disable_sn_now(self):
        self.disable_sn.assign(True)

# ---------- Spectral Normalized Conv2D ----------
class SNConv2D(tf.keras.layers.Layer):
    """
    Conv2D with spectral normalization on the kernel.
    Filter variable names match legacy ckpt:
      w: (kh, kw, in_channels, out_channels)
      u: (1, out_channels)
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='SAME',
                 power_iterations=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = int(filters)
        if isinstance(kernel_size, (tuple, list)):
            self.kh, self.kw = int(kernel_size[0]), int(kernel_size[1])
        else:
            self.kh = self.kw = int(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding.upper()
        self.power_iterations = int(power_iterations)
        self.disable_sn = tf.Variable(False, trainable=False, dtype=tf.bool, name="disable_sn")

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        var_dtype = tf.float32

        self.w = self.add_weight(
            name="w",
            shape=(self.kh, self.kw, in_ch, self.filters),
            initializer="glorot_uniform", dtype=var_dtype, trainable=True
        )
        self.u = self.add_weight(
            name="u", shape=(1, self.filters),
            initializer=tf.random_normal_initializer(), dtype=var_dtype, trainable=False
        )
        super().build(input_shape)

    def _spectral_norm(self, w):
        # reshape to [M, N] with N = out_channels
        w_mat = tf.reshape(w, [-1, self.filters])
        u = tf.stop_gradient(self.u)
        eps = 1e-12
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, w_mat, transpose_b=True), epsilon=eps)
            u = tf.nn.l2_normalize(tf.matmul(v, w_mat), epsilon=eps)
        sigma = tf.matmul(tf.matmul(v, w_mat), u, transpose_b=True)
        sigma = tf.stop_gradient(sigma)
        w_sn = tf.reshape(w_mat / (sigma + eps), tf.shape(w))
        return w_sn + 0.0 * (w - tf.stop_gradient(w))

    def call(self, x, training=False):
        x = tf.cast(x, self.w.dtype)
        w_use = tf.cond(self.disable_sn, lambda: self.w, lambda: self._spectral_norm(self.w))
        return tf.nn.conv2d(x, w_use, strides=[1, *self.strides, 1], padding=self.padding)

    def enable_sn(self, enabled=True):
        self.disable_sn.assign(not bool(enabled))

    def disable_sn_now(self):
        self.disable_sn.assign(True)

# ---------- Spectral Normalized Conv2DTranspose ----------
class SNConv2DTranspose(tf.keras.layers.Layer):
    """
    Conv2DTranspose (deconv) with spectral normalization.
    Filter variable names match legacy ckpt:
      w: (kh, kw, out_channels, in_channels)
      u: (1, out_channels)
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='SAME',
                 power_iterations=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = int(filters)
        if isinstance(kernel_size, (tuple, list)):
            self.kh, self.kw = int(kernel_size[0]), int(kernel_size[1])
        else:
            self.kh = self.kw = int(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding.upper()
        self.power_iterations = int(power_iterations)
        self.disable_sn = tf.Variable(False, trainable=False, dtype=tf.bool, name="disable_sn")

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        var_dtype = tf.float32

        # NOTE: conv2d_transpose filter: [kh, kw, out_ch, in_ch]
        self.w = self.add_weight(
            name="w",
            shape=(self.kh, self.kw, self.filters, in_ch),
            initializer="glorot_uniform", dtype=var_dtype, trainable=True
        )
        self.u = self.add_weight(
            name="u", shape=(1, self.filters),
            initializer=tf.random_normal_initializer(), dtype=var_dtype, trainable=False
        )
        super().build(input_shape)

    def _spectral_norm(self, w):
        # reshape to [M, N] with N = out_channels
        w_mat = tf.reshape(w, [-1, self.filters])
        u = tf.stop_gradient(self.u)
        eps = 1e-12
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, w_mat, transpose_b=True), epsilon=eps)
            u = tf.nn.l2_normalize(tf.matmul(v, w_mat), epsilon=eps)
        sigma = tf.matmul(tf.matmul(v, w_mat), u, transpose_b=True)
        sigma = tf.stop_gradient(sigma)
        w_sn = tf.reshape(w_mat / (sigma + eps), tf.shape(w))
        return w_sn + 0.0 * (w - tf.stop_gradient(w))

    def call(self, x, training=False):
        x = tf.cast(x, self.w.dtype)
        w_use = tf.cond(self.disable_sn, lambda: self.w, lambda: self._spectral_norm(self.w))
        b = tf.shape(x)[0]
        h = tf.shape(x)[1] * self.strides[0]
        w = tf.shape(x)[2] * self.strides[1]
        out_shape = tf.stack([b, h, w, self.filters])
        return tf.nn.conv2d_transpose(
            x, w_use, out_shape, strides=[1, *self.strides, 1], padding=self.padding
        )

    def enable_sn(self, enabled=True):
        self.disable_sn.assign(not bool(enabled))

    def disable_sn_now(self):
        self.disable_sn.assign(True)

#---------------- ResBlock layer Class-------------------
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, stride, name):
        super().__init__(name=name)
        # --- Add these two lines ---
        self.hidden_dim = hidden_dim
        self.stride = stride
        # ---------------------------

        self.deconv = SNConv2DTranspose(
            filters=hidden_dim, kernel_size=(1,3), strides=stride, 
            padding="SAME", name="deconv", dtype="float32"
        )
        self.bn = BatchNorm(name="bn", dtype="float32")
        self.act = tf.keras.layers.LeakyReLU(alpha=0.2, name="act", dtype="float32")

    def compute_output_shape(self, input_shape):
        """
        Calculates the output shape of the layer.
        """
        # Now self.stride and self.hidden_dim are correctly defined
        output_shape = [
            input_shape[0],
            input_shape[1] * self.stride[0],
            input_shape[2] * self.stride[1],
            self.hidden_dim 
        ]
        return tf.TensorShape(output_shape)
        
    def call(self, x, training=False):
        # Your call method is perfect as is.
        # Note: I'm assuming your SNConv2DTranspose handles the 'training' argument.
        # If not, it should be `self.deconv(x)`. But usually it's fine.
        x = self.deconv(x, training=training)
        x = self.bn(x, training=training)
        return self.act(x)
