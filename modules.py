import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, Dropout, Dense, Reshape
import numpy as np


def normalize(x, power=1):
    power_emp = tf.reduce_mean(tf.square(x))
    x = tf.sqrt(power / power_emp) * x
    return power_emp, x


class DepthToSpace(tf.keras.layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def call(self, x):
        return tf.nn.depth_to_space(x, self.bs)


def awgn(snr, x):
    # snr(db)
    n = 1 / (10 ** (snr / 10))
    sqrt_n = tf.sqrt(n)
    noise = tf.random.normal(shape=tf.shape(x)) * sqrt_n
    x_hat = x + noise
    return x_hat


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.prelu1 = PReLU()
        self.conv2 = Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                Conv2D(out_channels, kernel_size=1, strides=stride, use_bias=False),
                BatchNormalization()
            ])
        self.prelu2 = PReLU()

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.prelu2(out)
        return out


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.in_channels = 64
        self.conv1 = tf.keras.Sequential([
            Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False),
            BatchNormalization(),
            PReLU()
        ])
        self.layer1 = self._make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        if config.mod_method == 'bpsk':
            self.layer4 = self._make_layer(ResidualBlock, config.channel_use, 2, stride=2)
        else:
            self.layer4 = self._make_layer(ResidualBlock, config.channel_use * 2, 2, stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return tf.keras.Sequential(layers)

    def call(self, x):
        z0 = self.conv1(x)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return z4


class Decoder_Recon(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Decoder_Recon, self).__init__()
        self.config = config
        if config.mod_method == 'bpsk':
            input_channel = int(config.channel_use / (4 * 4))
        else:
            input_channel = int(config.channel_use * 2 / (4 * 4))

        self.conv1 = tf.keras.Sequential([
            Conv2D(256, kernel_size=1, strides=1, padding='valid'),
            PReLU()
        ])
        self.layer1 = tf.keras.Sequential([
            self._make_layer(ResidualBlock, 256, 2, stride=1),
            PReLU()
        ])
        self.layer2 = tf.keras.Sequential([
            self._make_layer(ResidualBlock, 256, 2, stride=1),
            PReLU()
        ])
        self.depth_to_space1 = DepthToSpace(4)
        self.conv2 = tf.keras.Sequential([
            Conv2D(128, kernel_size=1, strides=1, padding='valid'),
            PReLU()
        ])
        self.layer3 = tf.keras.Sequential([
            self._make_layer(ResidualBlock, 128, 2, stride=1),
            PReLU()
        ])
        self.depth_to_space2 = DepthToSpace(2)
        self.conv3 = Conv2D(3, kernel_size=1, strides=1, padding='valid')

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return tf.keras.Sequential(layers)

    def call(self, z):
        z0 = self.conv1(tf.reshape(z, [z.shape[0], -1, 4, 4]))
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.depth_to_space1(z2)
        z4 = self.conv2(z3)
        z5 = self.layer3(z4)
        z5 = self.depth_to_space2(z5)
        z6 = self.conv3(z5)
        return z6


class Decoder_Class(tf.keras.layers.Layer):
    def __init__(self, half_width, layer_width):
        super(Decoder_Class, self).__init__()
        self.layer_width = layer_width
        self.half_width = half_width
        self.fc1 = tf.keras.Sequential([
            Dropout(0.5),
            Dense(self.layer_width, activation=PReLU())
        ])
        self.fc2 = tf.keras.Sequential([
            Dropout(0.5),
            Dense(self.layer_width, activation=PReLU())
        ])
        self.fc3 = tf.keras.Sequential([
            Dropout(0.5),
            Dense(self.layer_width, activation=PReLU())
        ])
        self.fc4 = tf.keras.Sequential([
            Dropout(0.5),
            Dense(self.layer_width, activation=PReLU())
        ])
        self.last_fc = Dense(10)

    def call(self, z):
        x1 = self.fc1(z[:, :self.half_width])
        x2 = self.fc2(tf.concat([z[:, self.half_width:2 * self.half_width], x1], axis=1))
        x3 = self.fc3(tf.concat([z[:, :self.half_width], x2], axis=1))
        x4 = self.fc4(tf.concat([z[:, self.half_width:2 * self.half_width], x3], axis=1))
        x = tf.concat([x1, x2, x3, x4], axis=1)
        y_class = self.last_fc(x)
        return y_class
