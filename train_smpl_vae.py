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
torch.manual_seed(1234)
from pathlib import Path
from vae import BaseVAE, SMPLVAE, MyVAE
from dataset import SMPLightningData
from torch import Tensor
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


class SMPLexperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(SMPLexperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.h_in = self.params['h_in']
        self.h_out = self.params['h_out']
        self.curr_device = None
        self.hold_graph = False
        self.compress_loss = []
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        imu, gt = batch  # (b, 1, h_in), (b, 1, h_out)
        imu = torch.squeeze(imu)
        self.curr_device = imu.device
        results = self.forward(imu, labels=gt)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        imu, gt = batch
        imu = torch.squeeze(imu)
        self.curr_device = imu.device
        results = self.forward(imu, labels=gt)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_data()

    def get_mse(self, x, y):
        mse = ((x - y) ** 2).mean(axis=None)  # # (b, h_out)
        return mse

    def save_fig(self, filename):
        plt.savefig(filename)
        plt.close()

    def sample_data(self):
        # Get sample reconstruction data
        test_samples_list = list(iter(self.trainer.datamodule.test_dataloader()))
        nb_test_samples = len(test_samples_list)
        random_idx = np.random.choice(nb_test_samples)
        imu, gt = test_samples_list[random_idx]
        imu = torch.squeeze(imu)
        imu = imu.to(self.curr_device)
        gt = gt.to(self.curr_device)
        b = self.trainer.datamodule.val_batch_size

        recons = self.model.generate(imu, labels=gt)
        cs_loss = self.get_mse(recons.cpu().data, gt.cpu().data)
        self.compress_loss.append(cs_loss)

        fname = os.path.join(self.logger.log_dir, "Reconstructions",
                             f"{self.logger.name}_Epoch_{self.current_epoch}.png")

        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        # reshape  [b, 1, 9, 8] -> [b, 72]
        recons = np.reshape(recons.cpu().data, [b, 72])
        labels = np.reshape(gt.cpu().data, [b, 72])
        ax1.plot(recons[0, :], linestyle='--', label='Recons')
        ax1.plot(labels[0, :], linestyle='-', label='Labels')
        ax1.set_title('Real-time prediction')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('SMPL pose')
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


def run():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    config_file = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/smplvae.yaml'
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
    vae_models = {'SMPLVAE': SMPLVAE}
    model = vae_models[config['model_params']['name']](**config['model_params'])
    print(model)
    experiment = SMPLexperiment(model, config['exp_params'])
    data = SMPLightningData(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
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