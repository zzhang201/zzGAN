import tensorflow as tf
from tensorflow.keras import layers

class GumbelDiscriminator(tf.keras.Model):
    def __init__(self, flags):
        super(GumbelDiscriminator, self).__init__()
        self.config = flags
        self.seq_len = flags.sequence_length
        self.vocab_size = flags.vocab_size  # Typically 20 for amino acids

        # Define convolutional layers
        self.conv1 = layers.Conv2D(64, (1, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(128, (1, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, (1, 5), strides=(1, 1), padding='same', activation='relu')

        # Define dense layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.out = layers.Dense(1)  # Output logit for WGAN

    def call(self, inputs, training=False):
        """
        inputs: Tensor of shape [batch_size, sequence_length, vocab_size]
        """
        # Expand dimensions to add a channel dimension: [B, L, V] -> [B, L, V, 1]
        x = tf.expand_dims(inputs, axis=-1)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten and apply dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.out(x)

        return output
