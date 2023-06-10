import pickle as pkl
import numpy as np
from dataset import IMUDataset
from torch.utils.data import DataLoader
from torchvision import transforms


if __name__ == '__main__':
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    n_input = 867
    time_window = 60
    m_measurement = 20

    train_dataset = IMUDataset(file_path, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    train_examples = enumerate(train_data_loader)
    test_examples = enumerate(test_data_loader)

    test_len = len(list(iter(test_data_loader)))
    print('test_len: {}'.format(test_len))
    random_idx = np.random.choice(test_len)
    print('random element: {}'.format(list(iter(test_data_loader))[random_idx]))

    for e in train_examples:
        batch_idx, (imu_ori, imu_acc) = e
        print('index: {}, imu_ori.shape: {}, imu_acc.shape: {}'.format(batch_idx, imu_ori.shape, imu_acc.shape))
        for j in range(imu_ori.shape[0]):
            if np.isnan(imu_ori[j]).any():
                print(imu_ori[j])

    for e in test_examples:
        batch_idx, (imu_ori, imu_acc) = e
        print('index: {}, imu_ori_test.shape: {}, imu_acc_test.shape: {}'.format(batch_idx, imu_ori.shape, imu_acc.shape))
        for j in range(imu_ori.shape[0]):
            if np.isnan(imu_ori[j]).any():
                print(imu_ori[j])



