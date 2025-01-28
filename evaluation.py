import tensorflow as tf
from tqdm import tqdm
from utils import count_percentage, PSNR, SSIM, MSE


def EVAL(model, data_loader, config, epoch=0):
    acc_total = 0
    mse_total = 0
    psnr_total = 0
    ssim_total = 0
    total = 0

    # Placeholder for constellation data
    z_total = tf.zeros((10000, config.channel_use * 2))

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        total += len(target)

        # Preprocess data and target
        data = tf.convert_to_tensor(data)
        target = tf.convert_to_tensor(target)

        # Forward pass
        code, z, z_hat, pred, rec = model(data, training=False)

        # Save the constellation
        if config.mode == 'test' and config.mod_method in ['16qam', '64qam']:
            if batch_idx <= int(10000 / config.batch_size) - 1:
                z_total[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size, :].assign(code)
            elif batch_idx == int(10000 / config.batch_size):
                z_total[int(10000 / config.batch_size) * config.batch_size:, :].assign(code)
                count_percentage(
                    z_total.numpy(), config.mod_method, -1, config.snr_train, config.channel_use, config.tradeoff_lambda
                )
        else:
            if batch_idx == 0:
                count_percentage(
                    code.numpy(), config.mod_method, epoch, config.snr_train, config.channel_use, config.tradeoff_lambda
                )

        # Calculate metrics
        acc = tf.reduce_sum(tf.cast(tf.argmax(pred, axis=1) == tf.cast(target, tf.int64), tf.float32))
        mse = MSE(data, rec)
        psnr = PSNR(data, rec)
        ssim = SSIM(data, rec)

        acc_total += acc
        mse_total += mse
        psnr_total += psnr
        ssim_total += ssim

    acc_total /= total
    mse_total /= total
    psnr_total /= total
    ssim_total /= total

    return acc_total.numpy(), mse_total.numpy(), psnr_total.numpy(), ssim_total.numpy()
