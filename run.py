import os
import argparse
import yaml
import torchvision.utils as vutils
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from vae import VanillaVAE, BaseVAE, MyVAE
from dataset import VAEDataset, CompSensDataset
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
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # TODO: Normalize data here instead of dataloader
        imu_ori, imu_acc = batch  # (64, 51)
        self.curr_device = imu_ori.device
        A = 1 / self.m_measurement * torch.randn(self.n_input, self.m_measurement)
        noise = 0.1 * torch.randn(self.trainer.datamodule.train_batch_size, self.m_measurement)
        # (64, n) x (n, m) + (64, m) -> (64, m)
        y_batch = torch.matmul(imu_ori.cpu().data, A) + noise
        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        imu_ori, imu_acc = batch  # (64, 51)
        self.curr_device = imu_ori.device
        A = 1 / self.m_measurement * torch.randn(self.n_input, self.m_measurement)
        noise = 0.1 * torch.randn(self.trainer.datamodule.val_batch_size, self.m_measurement)
        # (64, n) x (n, m) + (64, m) -> (64, m)
        y_batch = torch.matmul(imu_ori.cpu().data, A) + noise
        y_batch = y_batch.to(self.curr_device)

        results = self.forward(y_batch)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_data()

    def sample_data(self):
        # Get sample reconstruction data
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        A = torch.randn(self.n_input, self.m_measurement)
        noise = 0.1 * torch.randn(self.trainer.datamodule.val_batch_size, self.m_measurement)
        # (64, n) x (n, m) + (64, m) -> (64, m)
        y_batch = torch.matmul(test_input.cpu().data, A) + noise
        y_batch = y_batch.to(self.curr_device)

        # test_input, test_label = batch
        recons = self.model.generate(y_batch)
        # print('recons: {}'.format(recons))
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=2)

        try:
            samples = self.model.sample(64,
                                        self.curr_device,)  # shape (3*17*17,)
            # print('samples: {}'.format(samples))
            # Reshape data to the original format
            # samples = np.reshape(samples.cpu().data, [self.time_window, 17 * 3])
            # # Denormalize the data
            # scaler = self.trainer.datamodule.test_dataloader().dataset.scaler
            # samples = scaler.inverse_transform(samples)  # shape ([seq_len, 17*3])
            #
            # samples = np.reshape(samples, [self.time_window, 17, 3])
            #
            # if self.current_epoch % 3 == 0:
            #
            #     plt.plot(samples[:, 0, 0])
            #     plt.xlabel('Frame')
            #     plt.ylabel('IMU reading')
            #     plt.legend()
            #     plt.savefig(os.path.join(self.logger.log_dir,
            #                                    "Samples",
            #                                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),)
            #     plt.close()

            # vutils.save_image(samples.cpu().data,
            #                   os.path.join(self.logger.log_dir,
            #                                "Samples",
            #                                f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
            #                   normalize=True,
            #                   nrow=2)
        except Warning:
            pass

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