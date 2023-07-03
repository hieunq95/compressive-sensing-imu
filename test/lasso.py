import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from dataset import IMUDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import Lasso
from scipy import ndimage, sparse
from imu_utils import get_l2_norm


def loss_fn(A, Y, z):
    """
    :param A: (m, n)
    :param Y: (b, m)
    :param z: (b, n)
    :return:
    """
    loss = A @ z.T - Y.T  # (m, n) * (n, b) - (m, b) -> (m, b)
    loss = loss.T  # (b, m)
    return cp.norm2(loss) ** 2


def regularizer(z):
    return cp.norm1(z)


def objective_fn(A, Y, z, lambd):
    return loss_fn(A, Y, z) + lambd * regularizer(z)


def mse(A, Y, z):
    return (1.0 / Y.shape[0] * loss_fn(A, Y, z)).value


def generate_data(m=20, n=100, b=60):
    """ Return matrix and IMU samples"""
    np.random.seed(1234)
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'

    train_dataset = IMUDataset(file_path, mode='train', transform=None)
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    seq_len_train = len(train_dataset)
    seq_len_test = len(test_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=seq_len_train, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=seq_len_test, shuffle=False, drop_last=True)
    label_train, _ = list(iter(train_data_loader))[0]  # (seq_len, 1, n)
    label_test, _ = list(iter(test_data_loader))[0]  # (seq_len, 1, n)

    A = np.random.randn(m, n)  # (m, n)
    Y_train = np.matmul(A, np.squeeze(label_train).T)  # (m, n) * (n, b) -> (m, b)
    Y_train = Y_train.T  # (b, m)
    Y_test = np.matmul(A, np.squeeze(label_test).T)  # (m, n) * (n, b) -> (m, b)
    Y_test = Y_test.T  # (b, m)

    return A, Y_train, Y_test, \
           np.squeeze(label_train).cpu().detach().numpy(), np.squeeze(label_test).cpu().detach().numpy()


def fit_model(b, n, A, Y_train, Y_test):
    z = cp.Variable(shape=[b, n])
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(objective_fn(A, Y_train, z, lambd)))

    lambd_values = np.logspace(-2, 3, 50)
    beta_values = []
    train_errors = []
    test_errors = []
    lambd.value = 0.1
    opt_value = problem.solve(verbose=True)
    train_errors.append(mse(A, Y_train, z))
    test_errors.append(mse(A, Y_test, z))
    beta_values.append(z.value)
    print('test_error: {}'.format(test_errors))
    print('beta: {}'.format(beta_values))
    print('Optimal value: {}'.format(opt_value))


def lasso_cvxpy():
    print('Installed solver: {}'.format(cp.installed_solvers()))
    m = 162
    n = 204
    btz = 60
    A, Y_train, Y_test, label_train, label_test = generate_data(m, n, btz)
    Y_train = np.squeeze(Y_train.numpy())
    print('A: {}'.format(A.shape))
    print('Y_train: {}'.format(Y_train.shape))
    print('Y_test: {}'.format(Y_test.shape))
    fit_model(btz, n, A, Y_train, Y_test)


def lasso_sklearn():
    m = 102
    n = 204
    btz = 60
    lasso = Lasso(alpha=0.01)
    A, Y_train, Y_test, label_train, label_test = generate_data(m, n, btz)
    print('A: {}'.format(A.shape))
    print('Y_train: {}'.format(Y_train.shape))  # (seq_len, m)
    print('Y_test: {}'.format(Y_test.shape))  # (seq_len, m)
    # proj = proj_operator @ data.ravel()[:, np.newaxis]
    y_batch = Y_train[:btz, :].T  # get small amount of samples
    lasso.fit(X=A, y=y_batch)
    y_test = Y_test[:btz, :]
    recons = lasso.predict(y_test.T)
    coef = lasso.coef_.reshape([btz, n])  # (b, n)
    labels = label_test[:btz, :]

    print('labels: {}'.format(labels.shape))

    plt.plot(recons.T[0], '--', label='Recons')
    plt.plot(labels[0], label='Labels')
    print('Recons loss: {}'.format(get_l2_norm(recons, labels)))
    plt.xlabel('Features')
    plt.ylabel('IMU reading')
    plt.legend()
    # print('coeff: {}'.format(lasso.coef_.reshape([btz, n])))
    plt.savefig('./lasso.png')
    # plt.show()


if __name__ == '__main__':
    lasso_sklearn()
    # lasso_cvxpy()

