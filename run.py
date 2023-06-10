import os
import argparse
import yaml
import threading
import torchvision.utils as vutils
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from vae import VanillaVAE, BaseVAE, MyVAE
from dataset import CompSensDataset
from torch import Tensor
from torch import optim
from torch.utils.data import RandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


class VAEXperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.m_measurement = self.params['m_measurement']
        self.n_input = self.params['n_input']
        self.time_window = self.params['time_window']
        self.curr_device = None
        self.hold_graph = False
        self.A = 1 / self.m_measurement * torch.randn(self.n_input, self.m_measurement)
        self.noise_std = 0.01
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # TODO: Normalize data here instead of dataloader
        imu_ori, imu_acc = batch  # (b, tw, n)
        imu_ori_data = imu_ori.cpu().data
        batch_size = self.trainer.datamodule.train_batch_size

        self.curr_device = imu_ori.device
        noise = self.noise_std * torch.randn(batch_size, self.m_measurement)
        # (b * tw, n) x (n, m) + (b * tw, m) -> (b * tw, m)
        y_batch = torch.matmul(imu_ori_data, self.A) + noise
        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch, A=self.A.to(self.curr_device))
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        imu_ori, imu_acc = batch  # (b, tw, n)
        imu_ori_data = imu_ori.cpu().data
        batch_size = self.trainer.datamodule.val_batch_size

        self.curr_device = imu_ori.device
        noise = self.noise_std * torch.randn(batch_size, self.m_measurement)
        # (b * tw, n) x (n, m) + (b * tw, m) -> (b * tw, m)
        y_batch = torch.matmul(imu_ori_data, self.A) + noise
        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch, A=self.A.to(self.curr_device))
        val_loss = self.model.loss_function(*results,
                                            M_N=batch_size,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_data()

    def save_fig(self, filename):
        plt.savefig(filename)
        plt.close()

    def sample_data(self):
        # Get sample reconstruction data
        test_samples_list = list(iter(self.trainer.datamodule.test_dataloader()))
        nb_test_samples = len(test_samples_list)
        random_idx = np.random.choice(nb_test_samples)
        test_imu_ori, test_imu_acc = test_samples_list[random_idx]
        test_imu_ori = test_imu_ori.to(self.curr_device)
        test_imu_acc = test_imu_acc.to(self.curr_device)

        # (b, n) x (n, m) -> (b, m)
        noise = self.noise_std * torch.randn(self.trainer.datamodule.val_batch_size, self.m_measurement)
        y_batch = torch.matmul(test_imu_ori.cpu().data, self.A) + noise
        y_batch = y_batch.to(self.curr_device)

        nb_imus = self.trainer.datamodule.train_dataloader().dataset.nb_imus
        sample_size = self.model.out_size
        sampling_rate = self.trainer.datamodule.train_dataloader().dataset.sampling_rate
        nb_samples = self.trainer.datamodule.val_batch_size

        recons = self.model.generate(y_batch, A=self.A)
        recons = np.reshape(recons.cpu().data, [nb_samples, int(sample_size / 3), 3])
        labels = np.reshape(test_imu_ori.cpu().data, [nb_samples, int(sample_size / 3), 3])
        fname = os.path.join(self.logger.log_dir, "Reconstructions",
                             f"{self.logger.name}_Epoch_{self.current_epoch}.png")

        plt.plot(recons[:, 7, 1], '--', label='Recons')
        plt.plot(labels[:, 7, 1], label='Labels')
        plt.xlabel('Frame')
        plt.ylabel('IMU reading')
        plt.title('Reconstruction')
        plt.legend()
        save_thread = threading.Thread(target=self.save_fig, args=(fname,))
        save_thread.start()

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return


def run():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'],)
    vae_models = {'VanillaVAE': MyVAE}
    model = vae_models[config['model_params']['name']](**config['model_params'])
    print(model)
    experiment = VAEXperiment(model, config['exp_params'])
    data = CompSensDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True)],
                     strategy=DDPPlugin(find_unused_parameters=False), **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)


if __name__ == '__main__':
    run()