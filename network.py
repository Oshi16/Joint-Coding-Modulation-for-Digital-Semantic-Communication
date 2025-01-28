import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, ReLU, Flatten
from modules import Encoder, Decoder_Recon, Decoder_Class, awgn, normalize, ResidualBlock  # Ensure these modules are TensorFlow-compatible


def modulation(logits, mod_method='bpsk'):
    # Gumbel softmax implementation
    discrete_code = tf.nn.softmax(logits / 1.5, axis=-1)
    discrete_code = tf.cast(tf.equal(discrete_code, tf.reduce_max(discrete_code, axis=-1, keepdims=True)), dtype=tf.float32)

    if mod_method == 'bpsk':
        output = discrete_code[:, :, 0] * -1 + discrete_code[:, :, 1] * 1

    elif mod_method == '4qam':
        const = tf.constant([1, -1], dtype=tf.float32)
        temp = discrete_code * const
        output = tf.reduce_sum(temp, axis=2)

    elif mod_method == '16qam':
        const = tf.constant([-3, -1, 1, 3], dtype=tf.float32)
        temp = discrete_code * const
        output = tf.reduce_sum(temp, axis=2)

    elif mod_method == '64qam':
        const = tf.constant([-7, -5, -3, -1, 1, 3, 5, 7], dtype=tf.float32)
        temp = discrete_code * const
        output = tf.reduce_sum(temp, axis=2)

    else:
        raise ValueError("Modulation method not defined.")

    return output


class JCM(tf.keras.Model):
    def __init__(self, config):
        super(JCM, self).__init__()
        self.config = config

        # Define the number of probability categories
        if self.config.mod_method == 'bpsk':
            self.num_category = 2
        elif self.config.mod_method == '4qam':
            self.num_category = 2
        elif self.config.mod_method == '16qam':
            self.num_category = 4
        elif self.config.mod_method == '64qam':
            self.num_category = 8
        else:
            raise ValueError("Unsupported modulation method.")

        self.encoder = Encoder(self.config)

        if config.mod_method == 'bpsk':
            self.prob_convs = tf.keras.Sequential([
                Dense(config.channel_use * self.num_category, activation='relu')
            ])
        else:
            self.prob_convs = tf.keras.Sequential([
                Dense(config.channel_use * 2 * self.num_category, activation='relu')
            ])

        self.decoder_recon = Decoder_Recon(self.config)

        if self.config.mod_method == 'bpsk':
            self.decoder_class = Decoder_Class(int(config.channel_use / 2), int(config.channel_use / 8))
        else:
            self.decoder_class = Decoder_Class(int(config.channel_use * 2 / 2), int(config.channel_use * 2 / 8))

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, Dense):
                initializer = tf.keras.initializers.GlorotNormal()
                layer.kernel.assign(initializer(layer.kernel.shape))

    def reparameterize(self, probs):
        mod_method = self.config.mod_method
        code = modulation(probs, mod_method)
        return code

    def call(self, x, training=False):
        # Encoding
        x_f = tf.reshape(self.encoder(x), [x.shape[0], -1])
        z = tf.reshape(self.prob_convs(x_f), [x.shape[0], -1, self.num_category])
        code = self.reparameterize(z)

        # Normalization
        power, z = normalize(code)

        # Add noise depending on mode
        if self.config.mode == 'train':
            z_hat = awgn(self.config.snr_train, z)
        elif self.config.mode == 'test':
            z_hat = awgn(self.config.snr_test, z)
        else:
            raise ValueError("Unsupported mode.")

        # Decoding
        recon = self.decoder_recon(z_hat)
        r_class = self.decoder_class(z_hat)

        return code, z, z_hat, r_class, recon
