# VAE implementation
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Any
from imu_utils import matmul_A


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.height = args[0]
        self.width = args[1]

    def forward(self, x):
        return x[:, :, :self.height, :self.width]


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class ConvoVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 h_in: int,
                 h_out: int,
                 **kwargs) -> None:
        super(ConvoVAE, self).__init__()

        self.latent_dim = latent_dim
        self.w_in = 12
        self.h_in = h_in // self.w_in
        self.w_out = 12
        self.h_out = h_out // self.w_out
        h_dims = [32, 64]
        # [b, 1, h_in, tw]
        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels, h_dims[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(h_dims[0], h_dims[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(h_dims[1], h_dims[1], kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(h_dims[1], h_dims[1], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Flatten(),
        )
        ytest = self.encoder(torch.randn((100, 1, self.h_in, self.w_in)))
        print('ytest: {}'.format(ytest.size()))
        resized_h_in = 3
        resized_w_in = 3

        self.fc_mu = torch.nn.Linear(h_dims[1] * resized_h_in * resized_w_in, self.latent_dim)
        self.fc_var = torch.nn.Linear(h_dims[1] * resized_h_in * resized_w_in, self.latent_dim)

        self.decoder = nn.Sequential(
            # upsample the input's size to output's size
            torch.nn.Linear(self.latent_dim, h_dims[1] * resized_h_in * resized_w_in),
            Reshape(-1, h_dims[1], resized_h_in, resized_w_in),
            nn.ConvTranspose2d(h_dims[1], h_dims[1], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(h_dims[1], h_dims[1], kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(h_dims[1], h_dims[0], kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(h_dims[0], 1, kernel_size=(3, 3), stride=(1, 1), padding=0),
            Trim(self.h_out, self.w_out),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        # input [b, c, h, w]
        x = self.encoder(input)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        A = kwargs['A']
        return [x_hat, input, mu, log_var, A]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]  # (b, 1, h_out, w_out)
        labels = args[1]
        mu = args[2]
        log_var = args[3]
        A = args[4]

        # print('recons.shape: {}, input.shape: {}, A.shape: {}'.format(recons.shape, input.shape, A.shape))

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons = matmul_A(recons, A, noise=None)
        recons_loss = F.mse_loss(recons, labels, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]


class SMPLVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 h_in: int,
                 h_out: int,
                 **kwargs) -> None:
        super(SMPLVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.h_in = h_in
        self.h_out = h_out
        self.h_dim = 512

        # [b, 1, 204] -> encoder(x)
        encoder_layers = [
            nn.Dropout(0.2),
            nn.Linear(self.h_in, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)  # x -> encoder
        self.fc_mu = nn.Linear(self.h_dim, self.latent_dim)  # encoder -> fc_mu
        self.fc_var = nn.Linear(self.h_dim, self.latent_dim)  # encoder -> fc_var

        # Define the decoder layers as a list of tuples, where each tuple contains the
        # layer type and its corresponding parameters.
        self.decoder_input = nn.Linear(self.latent_dim, self.h_dim)  # fc_mu, fc_var -> decoder_input
        decoder_layers = [
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.h_dim, self.h_out),
            # nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder_layers)  # decoder_input -> decoder -> x_hat

    def encode(self, input: Tensor) -> List[Tensor]:
        x = self.encoder(input)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        x_hat = self.decoder_input(z)
        x_hat = self.decoder(x_hat)
        return x_hat

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        labels = kwargs['labels']
        return [x_hat, mu, log_var, labels]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        mu = args[1]
        log_var = args[2]
        labels = args[3]

        # print('recons.shape: {}, labels.shape: {}'.format(recons.shape, labels.shape))

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, labels, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]


class MyVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 h_in: int,
                 h_out: int,
                 **kwargs) -> None:
        super(MyVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.h_in = h_in
        self.h_out = h_out
        self.h_dim = 64
        torch.manual_seed(1234)

        # Define the encoder layers as a list of tuples, where each tuple contains the
        # layer type and its corresponding parameters.
        encoder_layers = [
            # nn.Dropout(0.2),
            nn.Linear(self.h_in, self.h_dim),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)  # x -> encoder
        self.fc_mu = nn.Linear(self.h_dim, self.latent_dim)  # encoder -> fc_mu
        self.fc_var = nn.Linear(self.h_dim, self.latent_dim)  # encoder -> fc_var

        # Define the decoder layers as a list of tuples, where each tuple contains the
        # layer type and its corresponding parameters.
        # self.decoder_input = nn.Linear(self.latent_dim, self.h_dim)  # fc_mu, fc_var -> decoder_input
        decoder_layers = [
            nn.Linear(self.latent_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_out),
            nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder_layers)  # decoder_input -> decoder -> x_hat

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # result = self.decoder_input(z)
        result = self.decoder(z)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # Encode the input data to get the mean and log-variance of the latent distribution
        mu, log_var = self.encode(input)

        # Reparameterize the latent distribution and get the latent code z.
        z = self.reparameterize(mu, log_var)

        # Decode the latent code to get the reconstructed output.
        x_hat = self.decode(z)
        A = kwargs['A']

        # Return the reconstructed output, mean, and log-variance for use in the loss function.
        return [x_hat, input, mu, log_var, A]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:s_func
        :param kwargs:
        :return:
        """
        recons = args[0]  # (b, h_out)
        input = args[1]  # (b, h_in)
        mu = args[2]
        log_var = args[3]
        A = args[4]  # (h_in, h_out)

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        # print('recons: {}, input: {}, A: {}'.format(recons.size(), input.size(), A.size()))
        recons = matmul_A(recons, A, noise=None)
        recons_loss = F.mse_loss(recons, input, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        # loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]