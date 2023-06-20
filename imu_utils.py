import numpy as np
import cv2
import torch


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


def matmul_A(x, A, noise):
    """Calculate x*A + noise -> y  from h_out -> h_in for input"""
    b = x.shape[0]
    h_out = x.shape[2]
    tw = x.shape[3]
    n = h_out * tw
    # [b, 1, h_out, tw] -> [b, tw, h_out, 1]
    x = x.permute(0, 3, 2, 1)
    x = torch.reshape(x, [b, n])  # n total measurements
    # (b, n) x (n, m) + (b, m) -> (b, m)
    y = torch.matmul(x, A) + noise
    h_in = int(y.shape[1] / tw)
    # (b, m) -> (b, tw, h_in, 1)
    y = torch.reshape(y, [b, tw * h_in])
    y = torch.reshape(y, [b, tw, h_in, 1])
    y = y.permute(0, 3, 2, 1)

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
