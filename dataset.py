import os
import pickle as pkl
import numpy as np
import torch
from imu_utils import *
from typing import Union, List, Sequence, Optional
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CelebA, MNIST
from torchvision import transforms
from pytorch_lightning import LightningDataModule


# IMU dataset
class IMUDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        # Take IMU readings ad input
        # data['imu_ori']: A numpy-array of shape (seq_length, 17, 3, 3)
        # data['imu_acc']: A numpy-array of shape (seq_length, 17, 3)
        self.sampling_rate = 60  # Hz
        self.time_window = 17  # size of a data sample. The number of data samples = int(seq_len / time_window)
        self.transform = transform
        self.mode = mode
        if self.mode == 'train':
            subjects = [
                        's_01', 's_02',
                        's_03', 's_04',
                        's_05', 's_06',
                        's_07', 's_08'
                        ]
        else:
            subjects = ['s_09', 's_10']
        pkl_files = []
        for s in subjects:
            subject_path = os.path.join(data_path, 'DIP_IMU/{}/'.format(s))
            for f in os.listdir(subject_path):
                if f.endswith('.pkl'):
                    pkl_files.append(os.path.join(subject_path, f))

        print('Loading {} IMU readings from files: ... \n'.format(self.mode))
        self.imu_ori, self.imu_acc = self.__get_data_chunks(pkl_files)
        print('Dataset length: {}'.format(self.__len__()))
        print('One sample shape: {}'.format(self.imu_ori[0].shape))

    def __len__(self):
        # Return the number of chunks
        return len(self.imu_ori)

    def __getitem__(self, index):
        """ Return the orientation and acceleration of IMU reading.
        `index' is the chunk index between 0 and nb_chunks
        input shape: (N x C x H x W) (batch_size x nb_channel x height x width) (image)
        -> (N x 3 x 17 x seq_len) (batch_size x nb_sensor_dims x nb_sensor x seq_len) (imu)
        """
        imu_ori = torch.from_numpy(np.array(self.imu_ori[index], dtype=np.float32))
        imu_acc = torch.from_numpy(np.array(self.imu_acc[index], dtype=np.float32))
        imu_ori = np.transpose(imu_ori)
        imu_acc = np.transpose(imu_acc)

        # Flatten for MLP VAE:
        imu_ori = imu_ori.flatten()
        imu_acc = imu_acc.flatten()
        # print('imu_ori.shape: {}'.format(imu_ori.shape))
        # imu_ori = self.transform(np.array(self.imu_ori[index], dtype=np.float32))
        # imu_acc = self.transform(np.array(self.imu_acc[index], dtype=np.float32))

        return imu_ori, imu_acc

    def __get_data_chunks(self, pkl_files):
        imu_ori_out = []
        imu_acc_out = []
        for f in pkl_files:
            imu_ori_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_ori']
            imu_acc_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_acc']
            # Rotate axes of orientation data
            seq_len = imu_ori_data.shape[0]
            imu_ori_data = rot_matrix_to_aa(np.reshape(imu_ori_data, [seq_len, 17*9]))
            imu_ori_data = np.reshape(imu_ori_data, [seq_len, 17, 3])
            # Divide data into different chunks
            nb_chunks = int(seq_len / self.time_window)
            print('--- file: {}, nb_chunks: {}'.format(f, nb_chunks))
            # Normalize data
            # Reshape data so that we can fit into scaler
            imu_ori_data = np.reshape(imu_ori_data, [seq_len, 17*3])
            imu_acc_data = np.reshape(imu_acc_data, [seq_len, 17*3])
            scaler = MinMaxScaler().fit(imu_acc_data)
            imu_ori_data = scaler.transform(imu_ori_data)
            imu_acc_data = scaler.transform(imu_acc_data)

            for i in range(nb_chunks):
                chunk_i_ori = imu_ori_data[i * self.time_window: (i+1) * self.time_window]  # (time_window, 17*3)
                chunk_i_acc = imu_acc_data[i * self.time_window: (i+1) * self.time_window]
                imu_ori_out.append(chunk_i_ori)
                imu_acc_out.append(chunk_i_acc)

        return imu_ori_out, imu_acc_out


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class VAEDataset(LightningDataModule):
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
        #       =========================  CelebA Dataset  =========================

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(self.patch_size),
                                             transforms.ToTensor(), ])

        self.train_dataset = MyCelebA(  # MyCelebA
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,  # False for CelebA
        )

        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


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
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0,), std=(0.3081,))
        ])

        self.train_dataset = IMUDataset(self.data_dir, mode='train', transform=None)
        self.test_dataset = IMUDataset(self.data_dir, mode='test', transform=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
