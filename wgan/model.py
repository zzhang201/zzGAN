import tensorflow as tf

class WGAN:
    def __init__(self, data_handler, noise):
        """
        data_handler: instance of Protein class
        noise: tf.Tensor of shape [batch_size, z_dim]
        """
        self.data_handler = data_handler
        self.noise = noise

        # Use models and optimizers directly from Protein
        self.g_model = data_handler.g_model
        self.d_model = data_handler.d_model
        self.g_optim = data_handler.g_optim
        self.d_optim = data_handler.d_optim

        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
