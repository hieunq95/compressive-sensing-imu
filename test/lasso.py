import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from dataset import IMUDataset
from torch.utils.data import DataLoader


def loss_fn(X, Y, beta):
    """
    :param X: (m, n)
    :param Y: (b, m)
    :param beta: (b, n)
    :return:
    """
    return cp.norm2(X @ beta - Y)**2


def regularizer(beta):
    return cp.norm1(beta)


def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0] * loss_fn(X, Y, beta)).value


def generate_data(m=20, n=100, b=60):
    """ Return matrix and IMU samples"""
    np.random.seed(1234)
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'

    train_dataset = IMUDataset(file_path, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    train_data_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=b, shuffle=False, drop_last=True)
    seq_len_train = len(list(iter(train_data_loader)))
    seq_len_test = len(list(iter(test_data_loader)))
    Y_train, _ = list(iter(train_data_loader))[0]  # (seq_len, 1, n)
    Y_test, _ = list(iter(test_data_loader))[0]  # (seq_len, 1, n)

    X = np.random.randn(m, n)  # (m, n)
    Y_train = np.matmul(X, np.squeeze(Y_train).T)  # (m, n) * (n, b) -> (m, b)
    Y_train = Y_train.T  # (b, m)
    Y_test = np.matmul(X, np.squeeze(Y_test).T)  # (m, n) * (n, b) -> (m, b)
    Y_test = Y_test.T  # (b, m)

    return X, Y_train, Y_test


def fit_model(b, n , X_train, Y_train, X_test, Y_test):
    beta = cp.Variable((b, n))  # this is x in compressed sensing
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train.T, beta.T, lambd)))

    lambd_values = np.logspace(-2, 3, 50)
    beta_values = []
    train_errors = []
    test_errors = []
    lambd.value = lambd_values[0]
    opt_value = problem.solve()
    train_errors.append(mse(X_train, Y_train.T, beta.T))
    test_errors.append(mse(X_test, Y_test.T, beta.T))
    beta_values.append(beta.value)
    print('test_error: {}'.format(test_errors))
    print('beta: {}'.format(beta_values))
    print('Optimal value: {}'.format(opt_value))


if __name__ == '__main__':
    m = 190
    n = 204
    btz = 60
    X, Y_train, Y_test = generate_data(m, n, btz)
    Y_train = np.squeeze(Y_train.numpy())
    print('X: {}'.format(X.shape))
    print('Y_train: {}'.format(Y_train.shape))
    print('Y_test: {}'.format(Y_test.shape))
    fit_model(btz, n, X, Y_train, X, Y_test)


