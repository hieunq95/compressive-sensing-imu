import os
import shutil
import argparse
import math
import yaml
import threading
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from vae import BaseVAE, MyVAE
from imu_utils import matmul_A
from dataset import CompSensDataset, IMUDataset
from torch import Tensor
from torch import optim
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
        self.h_in = self.params['h_in']
        self.h_out = self.params['h_out']
        self.curr_device = None
        self.hold_graph = False
        self.P_T = self.params['P_T']
        self.noise_std = self.params['eta']
        self.A, _ = self.get_A(d=2)
        self.compress_loss = []
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def get_A(self, d=2, varepsilon=1e-6, use_sigma_x=True):
        # Generate matrix A follows the proposition
        imu_all_set = IMUDataset('/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/', mode='train', transform=None)
        imu_all = torch.from_numpy(imu_all_set.imu)
        imu_all = torch.flatten(torch.squeeze(imu_all))
        if use_sigma_x:
            # We get the measurement matrix based on statistical results of training set
            sigma_x = torch.std(imu_all)
            mean_x = torch.mean(imu_all)
        else:
            # In case we do not use the statistical results from training set, let's keep these values fixed
            sigma_x = 0.2
            mean_x = 0

        bound_x2 = max((-1 * d * sigma_x + mean_x) ** 2, (d * sigma_x + mean_x) ** 2)
        sigma_a = math.sqrt(self.P_T - varepsilon) / (self.h_out * d * math.sqrt(bound_x2))
        print('sigma_x: {}, sigma_a: {}, mean_x: {}'.format(sigma_x, sigma_a, mean_x))
        return torch.normal(mean=0, std=sigma_a, size=[self.h_in, self.h_out]), sigma_a  # (m, n)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        imu, _ = batch  # (b, 1, h_out)
        batch_size = self.trainer.datamodule.train_batch_size

        self.curr_device = imu.device
        noise = torch.normal(mean=0, std=self.noise_std, size=[batch_size, self.h_in])
        imu_flat = torch.squeeze(imu)
        y_batch = matmul_A(imu_flat, self.A.to(self.curr_device), noise.to(self.curr_device))  # y = Ax + eta
        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch, A=self.A.to(self.curr_device))
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              g_z=self.params['gz_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        imu, gt = batch
        batch_size = self.trainer.datamodule.val_batch_size

        self.curr_device = imu.device
        imu_flat = torch.squeeze(imu)
        noise = torch.normal(mean=0, std=self.noise_std, size=[batch_size, self.h_in])
        y_batch = matmul_A(imu_flat, self.A.to(self.curr_device), noise.to(self.curr_device))
        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch, A=self.A.to(self.curr_device))
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],
                                            g_z=self.params['gz_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_data()

    def sample_data(self):
        # Get sample reconstruction data
        test_samples_list = list(iter(self.trainer.datamodule.test_dataloader()))
        nb_test_samples = len(test_samples_list)
        random_idx = np.random.choice(nb_test_samples)
        imu, _ = test_samples_list[random_idx]  # [b, 1, h_out]
        batch_size = self.trainer.datamodule.val_batch_size

        imu_flat = torch.squeeze(imu)
        noise = torch.normal(mean=0, std=self.noise_std, size=[batch_size, self.h_in])
        y_batch = matmul_A(imu_flat.to(self.curr_device), self.A.to(self.curr_device), noise.to(self.curr_device))
        y_batch = y_batch.to(self.curr_device)

        recons = self.model.generate(y_batch, A=self.A.to(self.curr_device))
        recons = recons.cpu().data
        labels = torch.squeeze(imu).cpu().data

        cs_loss = self.get_mse(recons, labels)
        self.compress_loss.append(cs_loss)

        fname = os.path.join(self.logger.log_dir, "Reconstructions",
                             f"{self.logger.name}_Epoch_{self.current_epoch}.png")

        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(recons[0, :], linestyle='--', label='Recons')
        ax1.plot(labels[0, :], linestyle='-', label='Labels')

        ax1.set_title('Real-time prediction')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('IMU reading')
        ax1.grid(linestyle='--')

        ax2.plot(self.compress_loss)
        ax2.set_title('Compress_loss on test data')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(linestyle='--')

        ax1.legend()
        ax2.legend()
        figure.tight_layout()

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

    def get_mse(self, x, y):
        mse = ((x - y)**2).mean(axis=None)  # # (b, h_out)
        return mse

    def save_fig(self, filename):
        plt.savefig(filename)
        plt.close()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)


def run():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    config_file = 'configs/vae.yaml'
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default=config_file)

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
    # Save config file and matrix A
    shutil.copyfile(config_file, os.path.join(tb_logger.log_dir, "config.yaml"))
    torch.save(experiment.A, os.path.join(tb_logger.log_dir, "A.pt"))

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)


if __name__ == '__main__':
    run()