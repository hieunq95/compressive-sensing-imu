import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import math
import numpy as np
import cv2
import torch


def get_mean_loss(x, y):
    l2_norm_array = np.power(np.subtract(x, y), 2)
    return np.mean(l2_norm_array)


def get_imu_positions(h_in):
    """" IMU map:
        'head:': 0,
        'spine2': 1,
        'belly': 2,
        'lchest': 3,
        'rchest': 4,
        'lshoulder': 5,
        'rshoulder': 6,
        'lelbow': 7,
        'relbow': 8,
        'lhip': 9,
        'rhip': 10,
        'lknee': 11,
        'rknee': 12,
        'lwrist': 13,
        'rwrist': 14,
        'lankle': 15,
        'rankle': 16
    """
    if h_in == 24:
        return [13, 14]
    elif h_in == 48:
        return [0, 1, 13, 14]
    elif h_in == 72:
        return [0, 1, 11, 12, 13, 14]
    elif h_in == 96:
        return [0, 1, 7, 8, 11, 12, 13, 14]
    elif h_in == 120:
        return [0, 1, 7, 8, 11, 12, 13, 14, 15, 16]
    elif h_in == 144:
        return [0, 1, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    elif h_in == 168:
        return [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    else:
        return [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


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
    """Calculate y = Ax + eta \
    :param  x (b, h_out)
    :param  A (h_in, h_out)
    :param  noise (b, h_in)
    :return - y (b, h_in)

    """
    # (h_in, h_out) * (h_out, b) + (h_in, b) -> (h_in, b)
    x_T = torch.transpose(x, 0, 1)  # (h_out, b)
    Ax_T = torch.mm(A, x_T)  # (h_in, b)
    if noise is not None:
        noise_T = torch.transpose(noise, 0, 1)  # (h_in, b)
        y = torch.add(Ax_T, noise_T)  # (h_in, b)
    else:
        y = Ax_T

    y_T = torch.transpose(y, 0, 1)

    return y_T


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


def plot_reconstruction_data(x, y, imu_start, imu_end, dir):
    # /home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/logs/ConvoVAE/version_37
    dir_save = os.path.join(dir, 'Evaluation')
    Path(dir_save).mkdir(exist_ok=True, parents=True)
    b = x.shape[0]  # [b, 1, h_out]

    # plot using matplotlib
    num_images = b
    for i in range(num_images):
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(x[i, imu_start*9:(imu_end+1)*9], linestyle='--', label='Recons')
        ax1.plot(y[i, imu_start*9:(imu_end+1)*9], linestyle='-', label='Labels')
        print('Measurement loss: {}'.format(get_mean_loss(x, y)))

        ax1.set_title('Real-time prediction')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('IMU reading')
        ax1.grid(linestyle='--')

        ax1.legend()
        figure.tight_layout()

        fname = os.path.join(dir_save, 'Evaluation_{}.png'.format(i))

        save_thread = threading.Thread(target=save_fig, args=(fname,))
        save_thread.start()
