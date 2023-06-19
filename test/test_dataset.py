import yaml
import numpy as np
from dataset import IMUDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    file_config = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/convvae.yaml'
    with open(file_config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    h_in = config['model_params']['h_in']
    h_out = config['model_params']['h_out']
    tw = config['model_params']['tw']
    batch_size = 64

    train_dataset = IMUDataset(file_path, tw=tw, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, tw=tw, mode='test', transform=None)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

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


