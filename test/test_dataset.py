import pickle as pkl
import numpy as np
from dataset import IMUDataset
from torch.utils.data import DataLoader
from torchvision import transforms


if __name__ == '__main__':
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    n_input = 34
    m_measurement = 10
    time_window = 12

    train_dataset = IMUDataset(file_path, time_window=time_window,
                                        conv_data=True, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, time_window=time_window,
                                       conv_data=True, mode='test', transform=None)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    train_examples = enumerate(train_data_loader)
    test_examples = enumerate(test_data_loader)

    train_len = len(list(iter(train_data_loader)))
    test_len = len(list(iter(test_data_loader)))
    # print('test_len: {}'.format(test_len))
    random_idx = np.random.choice(test_len)
    print('random element: {}'.format(list(iter(test_data_loader))[random_idx]))
    num_nan_train = 0
    num_nan_test = 0
    for e in train_examples:
        batch_idx, (imu, imu_ori, imu_acc) = e
        print('index: {}, imu.shape: {}, imu_ori.shape: {}, imu_acc.shape: {}'.format(
            batch_idx, imu.shape, imu_ori.shape, imu_acc.shape))
        for j in range(imu.shape[0]):
            if np.isnan(imu[j]).any():
                num_nan_train += 1

    for e in test_examples:
        batch_idx, (imu, imu_ori, imu_acc) = e
        print('index: {}, imu.shape: {}, imu_ori_test.shape: {}, imu_acc_test.shape: {}'.format(
            batch_idx, imu.shape, imu_ori.shape, imu_acc.shape))
        for j in range(imu.shape[0]):
            if np.isnan(imu[j]).any():
                num_nan_test += 1

    print('Num_nan_train: {}/{}'.format(num_nan_train, train_len))
    print('Num_nan_test: {}/{}'.format(num_nan_test, test_len))


