import math
import yaml
import numpy as np
import torch
import imu_utils
from dataset import IMUDataset
from torch.utils.data import DataLoader


def test_data_loader():
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    file_config = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/vae.yaml'
    with open(file_config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    h_in = config['model_params']['h_in']
    h_out = config['model_params']['h_out']
    batch_size = 64

    train_dataset = IMUDataset(file_path, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_examples = enumerate(train_data_loader)
    test_examples = enumerate(test_data_loader)

    train_len = len(train_dataset)
    test_len = len(test_dataset)
    num_nan_train = 0
    num_nan_test = 0
    for e in train_examples:
        batch_idx, (imu, gt) = e
        print('index: {}, imu.shape: {}, gt.shape: {}'.format(
            batch_idx, imu.shape, gt.shape))
        for j in range(imu.shape[0]):
            if np.isnan(imu[j]).any():
                num_nan_train += 1

    for e in test_examples:
        batch_idx, (imu, gt) = e
        print('index: {}, imu.shape: {}, gt.shape: {}'.format(
            batch_idx, imu.shape, gt.shape))
        for j in range(imu.shape[0]):
            if np.isnan(imu[j]).any():
                num_nan_test += 1

    print('Num_nan_train: {}/{}'.format(num_nan_train, train_len))
    print('Num_nan_test: {}/{}'.format(num_nan_test, test_len))


# Function to interpolate NaN values
def interpolate_nan_intervals(tensor, intervals):
    new_tensor = tensor.clone()
    for start_idx, end_idx in intervals:
        for feature_idx in range(tensor.shape[1]):
            for batch_idx in range(start_idx, end_idx + 1):
                if torch.isnan(new_tensor[batch_idx, feature_idx]):
                    prev_idx = batch_idx - 1
                    while prev_idx >= start_idx and torch.isnan(new_tensor[prev_idx, feature_idx]):
                        prev_idx -= 1
                    next_idx = batch_idx + 1
                    while next_idx <= end_idx and torch.isnan(new_tensor[next_idx, feature_idx]):
                        next_idx += 1

                    if prev_idx < start_idx or next_idx > end_idx:
                        continue  # Cannot interpolate at edges of the interval

                    prev_val = new_tensor[prev_idx, feature_idx]
                    next_val = new_tensor[next_idx, feature_idx]

                    alpha = (batch_idx - prev_idx) / (next_idx - prev_idx)
                    interpolated_val = prev_val + alpha * (next_val - prev_val)

                    new_tensor[batch_idx, feature_idx] = interpolated_val

    return new_tensor


def create_corrupted_tensor(data, indices_to_replace):
    new_tensor = data.clone()
    new_tensor[indices_to_replace] = np.nan
    return new_tensor


def test_power_normalization():
    batch_size = 10
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, mode='train', transform=None)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Matrix A
    P_T = 0.1
    h_in = 102
    h_out = 204
    varepsilon = 1e-6
    d = 2
    imu_all = torch.from_numpy(test_dataset.imu)
    imu_all = torch.flatten(torch.squeeze(imu_all))
    std_x = torch.std(imu_all)
    mean_x = torch.mean((imu_all))

    sigma_a2 = (P_T - varepsilon) / (h_in**2 * d**2 * (d * std_x + mean_x)**2)
    sigma_a = math.sqrt(sigma_a2)
    print('mean_x: {}, sigma_x: {}'.format(mean_x, std_x))
    print('sigma_a: {}'.format(sigma_a))

    A = torch.normal(mean=0, std=sigma_a, size=[h_in, h_out])  # (m, n)
    for epoch, (imu, _) in enumerate(test_data_loader):
        imu = torch.squeeze(imu)
        # print('x.size: {}'.format(imu.size()))
        y_batch = imu_utils.matmul_A(imu, A)
        # print('y.size: {}'.format(y_batch.size()))
        # print('||x||_2^2: {}'.format(torch.linalg.vector_norm(imu, ord=2, dim=1)))
        y_2_norm = torch.linalg.vector_norm(y_batch, ord=2, dim=1)
        if epoch % 10 == 0:
            print('1/m * ||y||_2^2: {}'.format(1/h_in * torch.square(y_2_norm)))

    print('example:')
    At = torch.asarray([[0.1, 0, 0.5], [-0.1, 0.2, 0.2]])  # 2 x 3
    x = torch.asarray([[1, 0, 0.5], [-1, -1, 0]])  # 2 * 3
    x = torch.transpose(x, 0, 1)
    print('At: {}, x: {}'.format(At, x.T))
    print('At*x: {}'.format(torch.matmul(At, x)))


def csnr_calculator(csnr_values=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], P=0.1):
    sigma_N = []
    for x in csnr_values:
        sigma_N.append(P / (10**(x / 10)))
    return sigma_N


if __name__ == '__main__':
    # test_data_loader()
    # test_power_normalization()
    print(csnr_calculator())




