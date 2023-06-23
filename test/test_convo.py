import torch
import yaml
from torch import nn
from vae import Reshape, Trim, SMPLVAE, ConvoVAE
from train_vae import VAEXperiment


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


def test_smpl_vae():
    fname = "/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/smplvae.yaml"
    with open(fname, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    x = torch.randn((64, 1, 17, 6))
    labels = torch.randn((61, 1, 9, 8))
    model = SMPLVAE(config['model_params']['in_channels'], config['model_params']['latent_dim'],
                    config['model_params']['h_in'], config['model_params']['h_out'])
    print(model)
    y = model(x, labels=labels)[0]
    print('x: {}, y: {}'.format(x.size(), y.size()))


def test_conv_vae():
    fname = "/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/convvae.yaml"
    with open(fname, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    h_in = config['model_params']['h_in'] // 6
    w_in = config['model_params']['h_in'] // h_in
    x = torch.randn((64, 1, h_in, w_in))

    model = ConvoVAE(config['model_params']['in_channels'], config['model_params']['latent_dim'],
                    config['model_params']['h_in'], config['model_params']['h_out'])
    print(model)
    exp = VAEXperiment(model, config['exp_params'])
    A = exp.A
    y = model(x, A=A)[0]
    print('x: {}, y: {}'.format(x.size(), y.size()))


if __name__ == '__main__':
    # x = torch.randn((64, 1, 12, 36))
    # test_nn(x)
    # test_smpl_vae()
    test_conv_vae()