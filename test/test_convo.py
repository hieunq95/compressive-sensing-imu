import torch
from torch import nn
from torch.nn import functional as F


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


def upsample_test():
    # With square kernels and equal stride
    # m = nn.ConvTranspose2d(16, 33, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    input = torch.randn(20, 16, 50, 100)
    output = m(input)
    print(output.size())
    # exact output size can be also specified as an argument
    input = torch.randn((1, 16, 12, 12))
    downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    h = downsample(input)
    print(h.size())
    output = upsample(h, output_size=[1, 16, 14, 14])
    print(output.size())


def reparameterize(z_mu, z_log_var):
    eps = torch.randn(z_mu.size(0), z_mu.size(1))
    z = z_mu + eps * torch.exp(z_log_var / 2.)
    return z


def test_nn(x):
    print('x: {}'.format(x.size()))
    encoder = nn.Sequential(
        nn.Conv2d(x.shape[1], 32, kernel_size=(2, 4), stride=(1, 1), padding=1),
        nn.LeakyReLU(0.01),
        nn.Conv2d(32, 64, kernel_size=(2, 4), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.01),
        nn.Conv2d(64, 64, kernel_size=(2, 4), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.01),
        nn.Conv2d(64, 64, kernel_size=(2, 4), stride=(1, 1), padding=1),
        nn.Flatten(),
    )
    y = encoder(x)
    print('y: {}'.format(y.size()))
    resized_height = 5
    resized_width = int(y.shape[-1] / (64 * resized_height))
    z_mean = torch.nn.Linear(64 * resized_height * resized_width, 2)
    z_logvar = torch.nn.Linear(64 * resized_height * resized_width, 2)

    decoder = nn.Sequential(
        torch.nn.Linear(2, 64 * resized_height * resized_width * 3),
        Reshape(-1, 64, resized_height, resized_width * 3),
        nn.ConvTranspose2d(64, 64, kernel_size=(2, 4), stride=(1, 1), padding=1),
        nn.LeakyReLU(0.01),
        nn.ConvTranspose2d(64, 64, kernel_size=(2, 4), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.01),
        nn.ConvTranspose2d(64, 32, kernel_size=(2, 4), stride=(2, 2), padding=0),
        nn.LeakyReLU(0.01),
        nn.ConvTranspose2d(32, 1, kernel_size=(2, 4), stride=(1, 1), padding=0),
        Trim(12, 51),
    )

    zmean, zlogvar = z_mean(y), z_logvar(y)
    print('zmean: {}, zlogvar: {}'.format(zmean.size(), zlogvar.size()))
    z = reparameterize(zmean, zlogvar)
    print('z: {}'.format(z.size()))

    x_hat = decoder(z)
    print('x_hat: {}'.format(x_hat.size()))


if __name__ == '__main__':
    x = torch.randn((64, 1, 12, 36))
    test_nn(x)
