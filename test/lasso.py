import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from dataset import IMUDataset
from torch.utils.data import DataLoader


def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0] * loss_fn(X, Y, beta)).value

def generate_data(n=51, m=30, sigma=5, density=0.2):
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'

    train_dataset = IMUDataset(file_path, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)

    train_examples = enumerate(train_data_loader)
    test_examples = enumerate(test_data_loader)

    A = 1 / m * np.random.randn(n, m)

    return A, train_examples, test_examples

def fit_model(m, n , X_train, Y_train):
    beta = cp.Variable(m)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

    lambd_values = np.logspace(-2, 3, 50)
    train_errors = []
    test_errors = []
    beta_values = []


