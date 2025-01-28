import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from evaluation import EVAL
from utils import save_checkpoint, PSNR


def train(config, net, train_iter, test_iter):
    learning_rate = config.lr
    epochs = config.train_iters

    # Separate learning rate for specific layers
    prob_convs_params = [var for var in net.prob_convs.trainable_variables]
    base_params = [var for var in net.trainable_variables if var not in prob_convs_params]

    optimizer = tf.keras.optimizers.Adam([
        {'params': base_params},
        {'params': prob_convs_params, 'lr': learning_rate / 2}], learning_rate)

    # Loss functions
    loss_f1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_f2 = tf.keras.losses.MeanSquaredError()
    results = {'epoch': [], 'acc': [], 'mse': [], 'psnr': [], 'ssim': [], 'loss': []}

    # Learning rate scheduler (CosineAnnealingWarmRestarts equivalent)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=config.train_iters + 1,
        t_mul=1.0,
        m_mul=1.0,
        alpha=1e-6
    )

    best_acc = 0
    for epoch in range(epochs):
        net.trainable = True
        epoch_loss = []
        acc_total_train = 0
        psnr_total_train = 0

        for X, Y in tqdm(train_iter):
            with tf.GradientTape() as tape:
                code, _, _, y_class, y_recon = net(X, training=True)

                loss_1 = loss_f1(Y, y_class)
                loss_2 = loss_f2(X, y_recon)

                loss = loss_1 + config.tradeoff_lambda * loss_2

            # Compute gradients and update weights
            gradients = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))

            epoch_loss.append(loss.numpy())

            # Accuracy & PSNR of the train set
            acc = tf.reduce_sum(tf.cast(tf.argmax(y_class, axis=1) == tf.cast(Y, tf.int64), tf.float32))
            acc_total_train += acc
            psnr = PSNR(X, y_recon)
            psnr_total_train += psnr

        # Update learning rate
        optimizer.learning_rate = lr_schedule(epoch)

        loss = sum(epoch_loss) / len(epoch_loss)
        acc_train = acc_total_train / 50000
        psnr_train = psnr_total_train / 50000

        # Evaluate the model
        acc, mse, psnr, ssim = EVAL(net, test_iter, config, epoch)
        print('epoch: {:d}, loss: {:.6f}, acc: {:.3f}, mse: {:.6f}, psnr: {:.3f}, ssim: {:.3f}, lr: {:.6f}'.format(
            epoch, loss, acc, mse, psnr, ssim, optimizer.learning_rate))
        print('train acc: {:.3f}'.format(acc_train))
        print('train psnr: {:.3f}'.format(psnr_train))

        acc_num = acc.numpy()
        results['epoch'].append(epoch)
        results['loss'].append(loss)
        results['acc'].append(acc_num)
        results['mse'].append(mse)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        # Save the best model
        if (epochs - epoch) <= 10 and acc_num > best_acc:
            file_name = os.path.join(config.model_path, config.mod_method)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            model_name = f'CIFAR_SNR{config.snr_train:.3f}_Trans{config.channel_use}_{config.mod_method}.h5'
            save_checkpoint(net, os.path.join(file_name, model_name))
            best_acc = acc_num

    # Save all the results
    data = pd.DataFrame(results)
    file_name = os.path.join(config.result_path, config.mod_method)
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    result_name = f'CIFAR_SNR{config.snr_train:.3f}_Trans{config.channel_use}_{config.mod_method}.csv'
    data.to_csv(os.path.join(file_name, result_name), index=False, header=False)
