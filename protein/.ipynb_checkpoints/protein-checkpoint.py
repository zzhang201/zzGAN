import tensorflow as tf
from gan.sngan.generator_gumbel import GumbelGenerator
from gan.wgan.discriminator import DiscriminatorModel
from tensorflow.keras.mixed_precision import LossScaleOptimizer
NUM_AMINO_ACIDS = 21

class Protein(object):
    def __init__(self, flags, logdir):
        self.config = flags
        # self.classes = properties["class_mapping"]
        # self.num_classes = len(self.classes)
        self.width = flags.seq_length
        self.shape = [flags.batch_size, 1, self.width, NUM_AMINO_ACIDS]


        # === Generator (Gumbel-based) ===
        self.g_model = GumbelGenerator(flags, self.shape)

        # === Discriminator (WGAN) ===
        self.d_model = DiscriminatorModel(flags.df_dim)

        # Base optimizers
        base_g_optim = tf.keras.optimizers.Adam(
            learning_rate=flags.generator_learning_rate,
            beta_1=flags.beta1,
            beta_2=flags.beta2
        )
        base_d_optim = tf.keras.optimizers.Adam(
            learning_rate=flags.discriminator_learning_rate,
            beta_1=flags.beta1,
            beta_2=flags.beta2
        )
        
        # Wrap with loss scaling
        self.g_optim = LossScaleOptimizer(base_g_optim, dynamic=True)
        self.d_optim = LossScaleOptimizer(base_d_optim, dynamic=True)

        # === Step Counter ===
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.g_model.global_step = self.g_model._no_dependency(self.global_step)
