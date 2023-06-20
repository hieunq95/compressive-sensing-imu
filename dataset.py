import os
import pickle as pkl
import numpy as np
import torch
from imu_utils import *
from typing import Union, List, Sequence, Optional
# from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


# IMU dataset
class IMUDataset(Dataset):
    def __init__(self, data_path, tw=1, mode='train', transform=None):
        # data['imu_ori']: A numpy-array of shape (seq_length, 17, 3, 3)
        # data['imu_acc']: A numpy-array of shape (seq_length, 17, 3)
        self.sampling_rate = 60  # Hz
        self.nb_imus = 17
        self.tw = tw  # 0.1s
        self.transform = transform
        self.mode = mode
        # self.ori_scaler = MinMaxScaler()
        # self.acc_scaler = MinMaxScaler()
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
                        # 's_01',
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
        self.imu, self.gt = self.__get_data_chunks(pkl_files)
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
        imu_out = []
        gt_out = []
        for f in pkl_files:
            imu_ori_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_ori']
            imu_acc_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_acc']
            gt_data = pkl.load(open(f, 'rb'), encoding='latin1')['gt']  # [seq_len, 72]
            seq_len = imu_ori_data.shape[0]
            # reshape ori [seq_len, 51]
            imu_ori_data = rot_matrix_to_aa(np.reshape(imu_ori_data, [seq_len, self.nb_imus * 9]))
            # reshape acc [seq_len, 51]
            imu_acc_data = np.reshape(imu_acc_data, [seq_len, self.nb_imus * 3])
            # rescale acc
            imu_acc_data = imu_acc_data / 9.8
            # merge and clean ori + acc + gt
            imu_data = np.concatenate((imu_ori_data, imu_acc_data), axis=1)
            merged_data = np.concatenate((imu_data, gt_data), axis=1)  # [seq_len, 174]
            # count number of Nan entries
            nan_mask = np.isnan(merged_data)
            row_nan_mask = np.any(nan_mask, axis=1)
            num_nan = np.count_nonzero(row_nan_mask)
            # discard entries with Nan values
            print('-- discard {} Nan entries out of {} entries----'.format(num_nan, seq_len))
            clean_merged_data = merged_data[~np.isnan(merged_data).any(axis=1)]
            new_seq_len = seq_len - num_nan
            # split again
            clean_imu_data = clean_merged_data[:, :self.nb_imus * 2 * 3]  # [seq_len, 102]
            clean_gt_data = clean_merged_data[:, self.nb_imus * 2 * 3:]  # [seq_len, 72]
            # output data [1, h_in, tw]
            nb_chunks = int(new_seq_len / self.tw)
            print('--- file: {}, nb_chunks: {}, imu_data: {}, gt_data: {}'.format(
                f, nb_chunks, clean_imu_data.shape, clean_gt_data.shape))

            for i in range(nb_chunks - 1):
                imu_i = clean_imu_data[i * self.tw: (i + 1) * self.tw]  # [tw, 102]
                gt_i = clean_gt_data[i * self.tw: (i + 1) * self.tw]  # [tw, 72]
                # transpose [tw, h] -> [h, tw]
                imu_i = np.transpose(imu_i)
                gt_i = np.transpose(gt_i)
                # reshape [h, tw] -> [1, h, tw]
                imu_i = np.reshape(imu_i, [1, imu_i.shape[0], imu_i.shape[1]])
                gt_i = np.reshape(gt_i, [1, gt_i.shape[0], gt_i.shape[1]])
                # append
                imu_out.append(imu_i)
                gt_out.append(gt_i)

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
            tw: int = 1,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tw = tw

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = IMUDataset(self.data_dir, tw=self.tw, mode='train', transform=None)
        self.validate_dataset = IMUDataset(self.data_dir, tw=self.tw, mode='validate', transform=None)
        self.test_dataset = IMUDataset(self.data_dir, tw=self.tw, mode='test', transform=None)

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


class SMPLightningData(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            tw: int = 1,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tw = tw

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = IMUDataset(self.data_dir, tw=self.tw, mode='train', transform=None)
        self.validate_dataset = IMUDataset(self.data_dir, tw=self.tw, mode='validate', transform=None)
        self.test_dataset = IMUDataset(self.data_dir, tw=self.tw, mode='test', transform=None)

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
            shuffle=True,
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

