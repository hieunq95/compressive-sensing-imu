import os
import pickle as pkl
import numpy as np
import torch
from imu_utils import *
from typing import Union, List, Sequence, Optional
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


# IMU dataset
class IMUDataset(Dataset):
    def __init__(self, data_path, mode='train', processing=False, transform=None):
        # data['imu_ori']: A numpy-array of shape (seq_length, 17, 3, 3)
        # data['imu_acc']: A numpy-array of shape (seq_length, 17, 3)
        self.sampling_rate = 60  # Hz
        self.nb_imus = 17
        self.nb_imu_features = 12
        self.transform = transform
        self.mode = mode
        # self.ori_scaler = MinMaxScaler()
        self.acc_scaler = MaxAbsScaler()
        if self.mode == 'train':
            subjects = [
                        's_01',
                        's_02',
                        's_03', 's_04',
                        's_05', 's_06',
                        's_07', 's_08'
                        ]
        elif self.mode == 'validate':
            subjects = [
                        's_01',
                        # 's_03',
                        's_07',
            ]
        else:
            subjects = [
                        's_09',
                        's_10'
                        ]
        pkl_files = []
        for s in subjects:
            subject_path = os.path.join(data_path, 'DIP_IMU/{}/'.format(s))
            for f in os.listdir(subject_path):
                if f.endswith('.pkl'):
                    pkl_files.append(os.path.join(subject_path, f))

        print('Loading {} IMU readings from files: ... \n'.format(self.mode))
        if processing:
            self.imu, self.gt = self.__get_data_chunks(pkl_files)
            seq_len = len(self.imu)
            imu_reshape = np.squeeze(self.imu)
            ori_data = imu_reshape[:, :153]
            # normalize acceleration
            acc_data = imu_reshape[:, 153:]
            acc_data_abs = self.acc_scaler.fit_transform(acc_data)
            self.imu = np.concatenate((ori_data, acc_data_abs), axis=1)
            self.imu = np.reshape(self.imu, [seq_len, 1, 204])
            np.savez(os.path.join(data_path, 'processed_{}.npz'.format(mode)), imu=self.imu, gt=self.gt)
        else:
            self.imu = np.load(os.path.join(data_path, 'processed_{}.npz'.format(mode)))['imu']
            self.gt = np.load(os.path.join(data_path, 'processed_{}.npz'.format(mode)))['gt']
        print('{} dataset length: {}'.format(self.mode, self.__len__()))
        print('One imu/gt shapes: {}/{}'.format(self.imu[0].shape, self.gt[0].shape))

    def __len__(self):
        # Return the number of chunks
        return len(self.imu)

    def __getitem__(self, index):
        imu = torch.from_numpy(np.array(self.imu[index], dtype=np.float32))
        gt = torch.from_numpy(np.array(self.gt[index], dtype=np.float32))

        return imu, gt

    def __get_data_chunks(self, pkl_files):
        """
        Return the IMU dataset. The output format is a tuple of imu with shape (seq_len, 1, 204)
        and gt with shape (seq_len, 1, 72)
        """
        imu_out = []
        gt_out = []
        for f in pkl_files:
            imu_ori_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_ori']  # [seq_len, 17, 3, 3]
            imu_acc_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_acc']   # [seq_len, 17, 3]
            gt_data = pkl.load(open(f, 'rb'), encoding='latin1')['gt']  # [seq_len, 72]
            seq_len = imu_ori_data.shape[0]

            imu_ori_data = np.reshape(imu_ori_data, [seq_len, self.nb_imus * 9])
            imu_acc_data = np.reshape(imu_acc_data, [seq_len, self.nb_imus * 3])
            # print('One ori sample: {}'.format(imu_acc_data[0, :]))
            imu_data = np.concatenate((imu_ori_data, imu_acc_data), axis=1)
            merged_data = np.concatenate((imu_data, gt_data), axis=1)  # [seq_len, 276]
            # count number of Nan entries
            nan_mask = np.isnan(merged_data)
            row_nan_mask = np.any(nan_mask, axis=1)
            num_nan = np.count_nonzero(row_nan_mask)
            # discard entries with Nan values
            print('-- discard {} Nan entries out of {} entries----'.format(num_nan, seq_len))
            clean_merged_data = merged_data[~np.isnan(merged_data).any(axis=1)]
            new_seq_len = seq_len - num_nan
            # split again
            clean_imu_data = clean_merged_data[:, :self.nb_imus * self.nb_imu_features]  # [seq_len, 204]
            clean_gt_data = clean_merged_data[:, self.nb_imus * self.nb_imu_features:]  # [seq_len, 72]
            # output data [1, nb_imus, nb_imu_features]
            print('--- file: {}, seq_len: {}, imu_data: {}, gt_data: {}'.format(
                f, new_seq_len, clean_imu_data.shape, clean_gt_data.shape))

            for i in range(new_seq_len):
                # # reshape
                # imu_i = np.reshape(clean_imu_data[i], [1, self.nb_imus, self.nb_imu_features])
                # gt_i = np.reshape(clean_gt_data[i], [1, 9, 8])  # to use with CNN
                # append
                imu_out.append(clean_imu_data[i])  # [1, 204]
                gt_out.append(clean_gt_data[i])  # [1, 72]

        return imu_out, gt_out


class CompSensDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = IMUDataset(self.data_dir, mode='train', transform=None)
        self.validate_dataset = IMUDataset(self.data_dir, mode='validate', transform=None)
        self.test_dataset = IMUDataset(self.data_dir, mode='test', transform=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validate_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=self.pin_memory,
        )


class SMPLightningData(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = IMUDataset(self.data_dir, mode='train', transform=None)
        self.validate_dataset = IMUDataset(self.data_dir, mode='validate', transform=None)
        self.test_dataset = IMUDataset(self.data_dir, mode='test', transform=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validate_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

