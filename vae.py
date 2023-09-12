# VAE implementation
import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Any
from imu_utils import matmul_A


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


class DIPVAE(BaseVAE):
    def __init__(self, latent_dim, h_in, h_out, eta, P_T, **kwargs) -> None:
        super(DIPVAE, self).__init__()
        self.latent_dim = latent_dim
        self.h_in = h_in
        self.h_out = h_out
        self.noise_std = eta
        self.P_T = P_T
        self.h_dims = [64, 64]
        torch.manual_seed(123)

        encoder_layers = [
            nn.Dropout(0.25),
            nn.Linear(self.h_in, self.h_dims[0]),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)  # x -> encoder
        self.fc_mu = nn.Linear(self.h_dims[0], self.latent_dim)  # encoder -> fc_mu
        self.fc_var = nn.Linear(self.h_dims[0], self.latent_dim)  # encoder -> fc_var

        decoder_layers = [
            nn.Linear(self.latent_dim, self.h_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.h_dims[1], self.h_out),
            nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder_layers)  # decoder_input -> decoder -> x_hat

    def vector_reduction(self, x, positions, device):
        b = x.size()[0]
        x_ori = torch.reshape(x, [b, 17, 12]).to(device)
        y_ori = x_ori[:, positions.tolist(), :]  # [b, 6, 12]
        y = torch.reshape(y_ori, [b, self.h_in]).to(device)
        # Power normalize
        y_norm_i = torch.norm(y, p=2, dim=1).to(device)  # [1, b]
        power = math.sqrt(self.h_in * self.P_T)
        a = power / torch.reshape(y_norm_i, [b, 1]).to(device)
        a = a.repeat(1, self.h_in).to(device)
        y = a * y

        return x, y

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        positions = kwargs['positions']

        return [x_hat, input, mu, log_var, positions]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]  # (b, h_out)
        input = args[1]  # (b, h_in)
        mu = args[2]
        log_var = args[3]
        positions = args[4]

        kld_weight = kwargs['M_N']
        recons, reduced_recons = self.vector_reduction(recons, positions, recons.device)
        recons_loss = F.mse_loss(reduced_recons, input, reduction='sum')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
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
                 eta: float,
                 P_T: float,
                 **kwargs) -> None:
        super(MyVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.h_in = h_in
        self.h_out = h_out
        self.h_dims = [64, 64]
        self.noise_std = eta
        self.P_T = P_T
        torch.manual_seed(1234)

        encoder_layers = [
            nn.Dropout(0.25),
            nn.Linear(self.h_in, self.h_dims[0]),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)  # x -> encoder
        self.fc_mu = nn.Linear(self.h_dims[0], self.latent_dim)  # encoder -> fc_mu
        self.fc_var = nn.Linear(self.h_dims[0], self.latent_dim)  # encoder -> fc_var

        decoder_layers = [
            nn.Linear(self.latent_dim, self.h_dims[1]),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(self.h_dims[1], self.h_out),
            nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder_layers)  # decoder_input -> decoder -> x_hat

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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
        recons = args[0]  # (b, h_out)
        input = args[1]  # (b, h_in)
        mu = args[2]
        log_var = args[3]
        A = args[4]  # (h_in, h_out)

        gz_loss = torch.norm(recons, p=1)  # L_1 regularizer
        gz_weight = kwargs['g_z']
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons = matmul_A(recons, A)
        recons_loss = F.mse_loss(recons, input, reduction='sum')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss + gz_weight * gz_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]