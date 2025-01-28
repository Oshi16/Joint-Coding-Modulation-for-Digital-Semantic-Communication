import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from network import JCM  # TensorFlow version of JCM
from train import train  # TensorFlow version of train
from evaluation import EVAL  # TensorFlow version of EVAL
from utils import init_seeds  # TensorFlow-compatible utilities
import os
import argparse


def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


def main(config):
    # Initialize random seed
    init_seeds()

    # Prepare training & test data
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1.0 / 255.0,
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    train_generator = train_datagen.flow(x_train, y_train, batch_size=config.batch_size)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=config.batch_size)

    # Initialize model and optimizer
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Using device: {device}")
    net = JCM(config)

    # Load checkpoint if required
    if config.load_checkpoint:
        model_name = f'{config.model_path}/{config.mod_method}_SNR{config.snr_train:.3f}_Trans{config.channel_use}.h5'
        if os.path.exists(model_name):
            net.load_weights(model_name)
            print(f"Loaded checkpoint from {model_name}")
        else:
            print(f"No checkpoint found at {model_name}")

    # Train or test
    if config.mode == 'train':
        print(f"Training with the modulation scheme {config.mod_method}.")
        train(config, net, train_generator, test_generator)

    elif config.mode == 'test':
        print("Start Testing.")
        acc, mse, psnr, ssim = EVAL(net, test_generator, config)
        print(f'acc: {acc:.3f}, mse: {mse:.3f}, psnr: {psnr:.3f}, ssim: {ssim:.3f}')

    else:
        print("Wrong mode input!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--channel_use', type=int, default=128)
    """Available modulation methods:"""
    """bpsk, 4qam, 16qam, 64qam"""
    parser.add_argument('--mod_method', type=str, default='64qam')
    parser.add_argument('--load_checkpoint', type=int, default=1)

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snr_train', type=float, default=18)
    parser.add_argument('--snr_test', type=float, default=18)
    """The tradeoff hyperparameter lambda between two tasks"""
    parser.add_argument('--tradeoff_lambda', type=float, default=200)

    # Misc
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--dataset_path', type=str, default='./dataset')

    config = parser.parse_args()

    mischandler(config)
    main(config)
