import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import numpy as np
import cv2
import torch


def get_l2_norm(x, y):
    l2_norm_array = np.power(np.subtract(x, y), 2)
    return np.mean(l2_norm_array)


def rot_matrix_to_aa(data):
    """
    Converts the orientation data given in rotation matrices to angle axis representation. `data' is expected in format
    (seq_length, n*9). Returns an array of shape (seq_length, n*3)
    """
    seq_length, n_joints = data.shape[0], data.shape[1] // 9
    data_r = np.reshape(data, [seq_length, n_joints, 3, 3])
    data_c = np.zeros([seq_length, n_joints, 3])
    for i in range(seq_length):
        for j in range(n_joints):
            data_c[i, j] = np.ravel(cv2.Rodrigues(data_r[i, j])[0])
    return np.reshape(data_c, [seq_length, n_joints*3])


def matmul_A(x, A, noise=None):
    """Calculate x*A + noise -> y  from h_out -> h_in for input.
    :param  x (b, 1, h_out, w_out)
    :param  A (h_out * w_out, h_in * w_in)
    :param  noise (b, 1, h_in, w_in)
    :return - y (b, 1, h_in, w_in)

    """
    b = x.shape[0]
    h_out = x.shape[2]
    w_out = x.shape[3]
    if noise is not None:
        h_in = noise.shape[2]
        w_in = noise.shape[3]
    else:
        w_in = 6
        h_in = A.shape[1] // w_in

    n = h_out * w_out
    m = h_in * w_in
    x_flat = torch.reshape(x, [b, n])
    # (b, n) * (n, m) -> (b, m)
    y = torch.matmul(x_flat, A)
    # (b, m) + (b, m) -> (b, m)
    if noise is not None:
        noise_flat = torch.reshape(noise, [b, m])
        y = torch.add(y, noise_flat)
    # (b, m) -> (b, 1, h_in, w_win)
    y = torch.reshape(y, [b, 1, h_in, w_in])

    return y


def transpose_transform(x, y, a):
    """Calculate A*G(z) -> y  from h_out -> h_in in the loss calculation"""
    b = x.shape[0]
    h_out = x.shape[2]
    tw = x.shape[3]
    h_in = y.shape[2]
    A = a
    # transpose x = [b, 1, h_out, tw] -> [b, tw, h_out, 1]
    x = x.permute(0, 3, 2, 1)
    x = torch.reshape(x, [b, h_out * tw])  # (b, n)
    # (b, n) x (n, m) -> (b, m)
    z = torch.matmul(x, A)
    # (b, m) = (b, h_in * tw) -> (b, tw, h_in, 1)
    z = torch.reshape(z, [b, h_in * tw])
    z = torch.reshape(z, [b, tw, h_in, 1])
    # re-transpose (b, tw, h_in, 1) -> (b, 1 , h_in, tw)
    z = z.permute(0, 3, 2, 1)

    return z


def denormalize_acceleration(x):
    """Denormalize the output of the VAE from size (b, 1, h_out, tw) -> (b, 1, h_out/2: h_out/2*9.8, tw)"""
    x_np = x.cpu().data.detach().numpy()
    h_out = x.shape[2]
    x_np_ori = x_np[:, :, :int(h_out / 2), :]
    x_np_acc = x_np[:, :, int(h_out / 2):, :] * 9.8
    x_new = np.concatenate((x_np_ori, x_np_acc), axis=2)
    y = torch.from_numpy(x_new)

    return y


def save_fig(filename):
    plt.savefig(filename)
    plt.close()


def plot_reconstruction_data(x, y, dir):
    # /home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/logs/ConvoVAE/version_37
    dir_save = os.path.join(dir, 'Evaluation')
    Path(dir_save).mkdir(exist_ok=True, parents=True)
    b = x.shape[0]  # [b, 1, h_out]

    # plot using matplotlib
    num_images = b
    for i in range(num_images):
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(x[i, :], linestyle='--', label='Recons')
        ax1.plot(y[i, :], linestyle='-', label='Labels')
        print('Measurement loss: {}'.format(get_l2_norm(x, y)))

        ax1.set_title('Real-time prediction')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('IMU reading')
        ax1.grid(linestyle='--')

        ax1.legend()
        figure.tight_layout()

        fname = os.path.join(dir_save, 'Evaluation_{}.png'.format(i))

        save_thread = threading.Thread(target=save_fig, args=(fname,))
        save_thread.start()
