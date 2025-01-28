import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse


def init_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state, filename="my_checkpoint"):
    print("=>Saving checkpoint")
    tf.saved_model.save(state, filename)


def count_percentage(code, mod, epoch, snr, channel_use, tradeoff_h):
    if mod == '4qam' or mod == 'bpsk':
        pass
    else:
        code = tf.reshape(code, [-1, 2]).numpy()
        index = [i for i in range(len(code))]
        random.shuffle(index)
        code = code[index]

        if mod == '16qam':
            I_point = np.array([-3, -1, 1, 3])
            order = 16
        elif mod == '64qam':
            I_point = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            order = 64

        I, Q = np.meshgrid(I_point, I_point)
        map = np.stack((I.flatten(), Q.flatten()), axis=-1)
        per_s = []
        fig = plt.figure(dpi=300)
        ax = Axes3D(fig)
        for i in range(order):
            temp = np.sum(np.abs(code - map[i]), axis=1)
            num = np.count_nonzero(temp == 0)
            per = num / code.shape[0]
            per_s.append(per)
        per_s = np.array(per_s)
        height = np.zeros_like(per_s)
        width = depth = 0.3
        ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average')
        file_name = './cons_fig/' + '{}_{}_{}_{}'.format(mod, snr, channel_use, tradeoff_h)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        fig.savefig(file_name + '/{}'.format(epoch))
        plt.close()

        # additional scatter plot
        if mod == '64qam':
            fig = plt.figure(dpi=300)
            for k in range(order):
                plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='b')
            fig.savefig(file_name + '/scatter_{}'.format(epoch))
            plt.close()


def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
    origin = ((tensor_org + 1) / 2).numpy()
    trans = ((tensor_trans + 1) / 2).numpy()
    for i in range(trans.shape[0]):
        psnr = 0
        for j in range(trans.shape[1]):
            psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
            psnr += psnr_temp
        psnr /= 3
        total_psnr += psnr
    return total_psnr


def SSIM(tensor_org, tensor_trans):
    total_ssim = 0
    origin = tensor_org.numpy()
    trans = tensor_trans.numpy()
    for i in range(trans.shape[0]):
        ssim = 0
        for j in range(trans.shape[1]):
            ssim_temp = comp_ssim(origin[i, j, :, :], trans[i, j, :, :], data_range=1.0)
            ssim += ssim_temp
        ssim /= 3
        total_ssim += ssim
    return total_ssim


def MSE(tensor_org, tensor_trans):
    origin = ((tensor_org + 1) / 2).numpy()
    trans = ((tensor_trans + 1) / 2).numpy()
    mse = np.mean((origin - trans) ** 2)
    return mse * tensor_org.shape[0]
