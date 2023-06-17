import os
import argparse
import yaml
import threading
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from vae import ConvoVAE, BaseVAE, MyVAE
from dataset import CompSensDataset
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
        self.m_measurement = self.params['m_measurement']
        self.n_input = self.params['n_input']
        self.time_window = self.params['time_window']
        self.conv_data = self.params['conv_data']
        self.curr_device = None
        self.hold_graph = False
        if self.conv_data:
            m = self.m_measurement * self.time_window * 3
            n = self.n_input * self.time_window * 3
            self.A = 1 / m * torch.randn(n, m)
        else:
            self.A = 1 / self.m_measurement * torch.randn(self.n_input, self.m_measurement)
        self.noise_std = 0.01
        self.compress_loss = []
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # TODO: Normalize data here instead of dataloader
        imu, _, _ = batch  # (b, tw, n)
        imu_data = imu.cpu().data  # (b, tw, 3 * nb_imus * 2)
        batch_size = self.trainer.datamodule.train_batch_size

        self.curr_device = imu.device
        if self.conv_data:
            m = self.m_measurement * self.time_window * 3
            n = self.n_input * self.time_window * 3
            noise = self.noise_std * torch.randn(batch_size, m)
            imu_data = torch.reshape(imu_data, [batch_size, n])
        else:
            noise = self.noise_std * torch.randn(batch_size, self.m_measurement)
        # (b * tw, n) x (n, m) + (b * tw, m) -> (b * tw, m)
        y_batch = torch.matmul(self.normalize_data(imu_data, False), self.A) + noise
        if self.conv_data:
            y_batch = torch.reshape(y_batch, [batch_size, 1, self.time_window, self.m_measurement * 3])

        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch, A=self.A.to(self.curr_device))
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        imu, _, _ = batch  # (b, tw, n)
        imu_data = imu.cpu().data
        batch_size = self.trainer.datamodule.val_batch_size

        self.curr_device = imu.device
        if self.conv_data:
            m = self.m_measurement * self.time_window * 3
            n = self.n_input * self.time_window * 3
            noise = self.noise_std * torch.randn(batch_size, m)
            imu_data = torch.reshape(imu_data, [batch_size, n])
        else:
            noise = self.noise_std * torch.randn(batch_size, self.m_measurement)
        # (b * tw, n) x (n, m) + (b * tw, m) -> (b * tw, m)
        y_batch = torch.matmul(self.normalize_data(imu_data, False), self.A) + noise
        if self.conv_data:
            y_batch = torch.reshape(y_batch, [batch_size, 1, self.time_window, self.m_measurement * 3])

        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch, A=self.A.to(self.curr_device))
        val_loss = self.model.loss_function(*results,
                                            M_N=batch_size,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_data()

    def get_l2_norm(self, x, y):
        l2_norm_array = np.power(np.subtract(x, y), 2)
        return torch.mean(l2_norm_array)

    def save_fig(self, filename):
        plt.savefig(filename)
        plt.close()

    def normalize_data(self, x, normalize=True):
        if normalize:
            ori_scaler = self.trainer.datamodule.train_dataset.ori_scaler
            x_ = Tensor(ori_scaler.transform(x))
        else:
            x_ = x
        return x_

    def sample_data(self):
        # Get sample reconstruction data
        test_samples_list = list(iter(self.trainer.datamodule.test_dataloader()))
        nb_test_samples = len(test_samples_list)
        random_idx = np.random.choice(nb_test_samples)
        test_imu, _, _ = test_samples_list[random_idx]
        test_imu = test_imu.to(self.curr_device)
        batch_size = self.trainer.datamodule.val_batch_size

        # (b, n) x (n, m) -> (b, m)
        test_imu = self.normalize_data(test_imu.cpu().data, False)
        if self.conv_data:
            m = self.m_measurement * self.time_window * 3
            n = self.n_input * self.time_window * 3
            noise = self.noise_std * torch.randn(batch_size, m)
            test_imu = torch.reshape(test_imu, [batch_size, n])
        else:
            noise = self.noise_std * torch.randn(batch_size, self.m_measurement)
        # (b * tw, n) x (n, m) + (b * tw, m) -> (b * tw, m)
        y_batch = torch.matmul(self.normalize_data(test_imu, False), self.A) + noise
        if self.conv_data:
            y_batch = torch.reshape(y_batch, [batch_size, 1, self.time_window, self.m_measurement * 3])

        y_batch = y_batch.to(self.curr_device)

        # we have 17*2 = 34 virtual IMUs as we consider 17 IMUs for orientation and 17 IMUs for acceleration
        nb_imus = self.trainer.datamodule.train_dataloader().dataset.nb_imus * 2
        sample_size = self.model.out_size
        sampling_rate = self.trainer.datamodule.train_dataloader().dataset.sampling_rate
        nb_samples = self.trainer.datamodule.val_batch_size

        recons = self.model.generate(y_batch, A=self.A)
        if self.conv_data:
            recons = np.reshape(recons.cpu().data, [nb_samples, self.time_window, sample_size, 3])
            labels = np.reshape(test_imu.cpu().data, [nb_samples, self.time_window, sample_size, 3])
        else:
            recons = np.reshape(recons.cpu().data, [nb_samples, int(sample_size / 3), 3])
            labels = np.reshape(test_imu.cpu().data, [nb_samples, int(sample_size / 3), 3])

        cs_loss = self.get_l2_norm(recons, labels)
        self.compress_loss.append(cs_loss)

        fname = os.path.join(self.logger.log_dir, "Reconstructions",
                             f"{self.logger.name}_Epoch_{self.current_epoch}.png")

        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        if self.conv_data:
            recons_plot = np.reshape(recons, [1, nb_samples * self.time_window, nb_imus, 3])
            labels_plot = np.reshape(labels, [1, nb_samples * self.time_window, nb_imus, 3])
            ax1.plot(recons_plot[0, :120, 7, 1], linestyle='--', label='Recons')
            ax1.plot(labels_plot[0, :120, 7, 1], linestyle='-', label='Labels')
        else:
            ax1.plot(recons[:, 7, 1], linestyle='--', label='Recons')
            ax1.plot(labels[:, 7, 1], linestyle='-', label='Labels')
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


def run(convo=False):
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    config_file = 'configs/convvae.yaml' if convo else 'configs/vae.yaml'
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
    vae_models = {'ConvoVAE': ConvoVAE} if convo else {'VanillaVAE': MyVAE}
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
    run(convo=True)