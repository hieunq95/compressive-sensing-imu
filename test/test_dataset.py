import yaml
import numpy as np
import torch
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


if __name__ == '__main__':
    # test_data_loader()

    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_data_loader = DataLoader(test_dataset, batch_size=60, shuffle=False, drop_last=True)
    imu, _ = list(iter(test_data_loader))[0]
    imu = torch.squeeze(imu)
    print('data: {}'.format(imu))

    # List of indices of missing batches
    missing_batch_indices = [0, 1, 2, 3, 4, 5, 54, 55, 56, 57, 58, 59]
    corrupted_imu = create_corrupted_tensor(imu, missing_batch_indices)
    print('new_data: {}'.format(corrupted_imu.size()))
    print(corrupted_imu)

    missing_intervals = [(0, 5), (54, 59)]
    interpolated_imu = interpolate_nan_intervals(corrupted_imu, missing_intervals)
    print(interpolated_imu, interpolated_imu.size())


