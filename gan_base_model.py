import tensorflow as tf

class GAN(object):
    def __init__(self, data_handler, noise):
        self.config = self.init_param()
        self.data_handler = data_handler
        self.dataset = self.config.dataset
        self.z_dim = self.config.z_dim
        self.gf_dim = self.config.gf_dim
        self.df_dim = self.config.df_dim
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.noise = noise

        # Load Generator and Discriminator
        self.discriminator, self.generator = self.get_discriminator_and_generator()

        # Build models
        self.build_model()

    def init_param(self):
        from absl import flags
        return flags.FLAGS

    def build_model(self):
        self.d_learning_rate, self.g_learning_rate = self.get_learning_rates()
        self.build_model_single_gpu(batch_size=self.config.batch_size)
        self.d_optim, self.g_optim = self.get_optimizers()

    def build_model_single_gpu(self, batch_size=1):
        batch = self.data_handler.get_batch(batch_size, self.config)
        real_x, labels = batch[0], batch[1]
        real_x, labels = self.data_handler.prepare_real_data(real_x, labels)
        self.fake_x = self.get_generated_data(self.noise, labels)

        self.discriminator_real = self.get_discriminator_result(real_x, labels)
        self.discriminator_fake = self.get_discriminator_result(self.fake_x, labels)

    def get_optimizers(self):
        d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.d_learning_rate,
            beta_1=self.config.beta1,
            beta_2=self.config.beta2,
            name='d_optimizer'
        )
        g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.g_learning_rate,
            beta_1=self.config.beta1,
            beta_2=self.config.beta2,
            name='g_optimizer'
        )
        return d_optimizer, g_optimizer

    def get_learning_rates(self):
        return self.config.discriminator_learning_rate, self.config.generator_learning_rate

    def get_discriminator_result(self, data, labels):
        return self.discriminator.discriminate(data, labels, training=True)

    def get_generated_data(self, noise, labels):
        return self.generator.generate(noise, labels)

    def get_discriminator_and_generator(self):
        raise NotImplementedError  # Your WGAN model should implement this

    def increment_step(self):
        self.global_step.assign_add(1)
