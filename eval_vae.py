import os
import time
import yaml
import smplx
import pyrender
import trimesh
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F
from train_smpl_vae import SMPLexperiment
from vae import SMPLVAE, MyVAE, DIPVAE
from train_mlp_vae import VAEXperiment
from train_dip import DIPExperiment
from dataset import IMUDataset
from imu_utils import matmul_A, plot_reconstruction_data, get_l2_norm, get_imu_positions
from sklearn.linear_model import Lasso


def measure_loss(x, y):
    loss = F.mse_loss(x, y, reduction='sum')
    return loss


def model_forward(model, pose, batch_size):
    global_orient = pose[:, :3].reshape(batch_size, 3)
    body_pose = pose[:, 3:].reshape(batch_size, 69)
    print('body_pose.shape: {}'.format(body_pose.shape))
    res = model(global_orient=global_orient, body_pose=body_pose)
    vertices = res.vertices.detach().cpu().numpy().squeeze()
    joints = res.joints.detach().cpu().numpy().squeeze()
    # print('vertices: {}'.format(vertices))

    return vertices, joints


def clear_scene(scene):
    """
    Update the scene and return a new node
    """
    for node in scene.get_nodes(name='joints'):
        if node.name is not None:
            scene.remove_node(node)
    for node in scene.get_nodes(name='mesh'):
        if node.name is not None:
            scene.remove_node(node)
    return scene


def body_from_vertices(vertices, faces, animation=False):
    # vertices.shape: (1075, 6890, 3)
    seq_len = vertices.shape[0]
    scene = pyrender.Scene()
    tri_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh, name='mesh')
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

    # Animate scene
    i = 0
    t = 0
    fps = 1
    while viewer.is_active:
        if t >= np.floor((seq_len - 1) / fps):
            t = 0
        else:
            t += 1
        if animation:
            tri_mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            viewer.render_lock.acquire()
            scene = clear_scene(scene)
            scene.add(mesh, name='mesh')
            viewer.render_lock.release()

            i += 1
        else:
            tri_mesh = trimesh.Trimesh(vertices=vertices[i], faces=faces)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            viewer.render_lock.acquire()
            scene = clear_scene(scene)
            scene.add(mesh, name='mesh')
            viewer.render_lock.release()

            # i += 0.01


def smpl_forward(imu, gt, vae_model, body_model, batch_size, animation=False):
    pose = vae_model(imu, labels=gt)[0]  # [b, 1, 72]

    faces = body_model.faces
    batsz = pose.shape[0]
    vts, jts = model_forward(body_model, pose, batsz)
    # Visualize
    body_from_vertices(vts, faces, animation)


def eval_models(vae_ver=0, spml_vae_ver=0, batch_size=64, batch_id=0, animation=False):
    smpl_vae_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/smplvae.yaml'
    vae_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
                       'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    A_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
                       'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)

    with open(smpl_vae_config_fname, 'r') as f_1:
        try:
            config_smpl_vae = yaml.safe_load(f_1)
        except yaml.YAMLError as exc:
            print(exc)

    with open(vae_config_fname, 'r') as f_2:
        try:
            config_vae = yaml.safe_load(f_2)
        except yaml.YAMLError as exc:
            print(exc)

    # load body model
    bm_fname = '/home/hinguyen/Data/smpl/models/smpl_male.pkl'
    body_model = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    print('Model: {}'.format(body_model))

    # load SMPL_VAE model
    smpl_vae_model = SMPLVAE(**config_smpl_vae['model_params'])
    smpl_trained_fname = os.path.join(config_smpl_vae['logging_params']['save_dir'],
                                      config_smpl_vae['model_params']['name'], 'version_{}'.format(spml_vae_ver),
                                      'checkpoints', 'last.ckpt')

    exp1 = SMPLexperiment(smpl_vae_model, config_smpl_vae['exp_params'])
    smpl_vae_model = exp1.load_from_checkpoint(smpl_trained_fname, vae_model=smpl_vae_model,
                                               params=config_smpl_vae['exp_params'])
    smpl_vae_model.eval()

    # load VAE model
    vae_model = MyVAE(**config_vae['model_params'])
    vae_trained_fname = os.path.join(config_vae['logging_params']['save_dir'],
                                      config_vae['model_params']['name'], 'version_{}'.format(vae_ver),
                                      'checkpoints', 'last.ckpt')
    exp2 = VAEXperiment(vae_model, config_vae['exp_params'])
    vae_model = exp2.load_from_checkpoint(vae_trained_fname, vae_model=vae_model,
                                            params=config_vae['exp_params'])
    vae_model.eval()

    # test model with test dataset
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_len = len(list(iter(test_loader)))
    print('Number of batches: {}'.format(test_len))
    e = list(iter(test_loader))[batch_id]

    # e = next(iter(test_examples))
    (imu, gt) = e

    # test VAE model with reconstructed data
    A = torch.load(A_fname)
    print('A: {}'.format(A))
    m = exp2.h_in
    imu_flat = torch.squeeze(imu)
    noise = torch.normal(mean=0, std=exp2.noise_std, size=[batch_size, m])
    y_batch = matmul_A(imu_flat, A, noise)
    recons = vae_model(y_batch, A=A)[0]  # [b, h_out]
    samples = vae_model.model.sample(batch_size, vae_model.curr_device)
    rand_samples = torch.randn(size=[batch_size, exp2.h_out])

    smpl_forward(rand_samples, torch.squeeze(gt), smpl_vae_model, body_model, batch_size, animation)
    smpl_forward(samples, torch.squeeze(gt), smpl_vae_model, body_model, batch_size, animation)
    smpl_forward(recons, torch.squeeze(gt), smpl_vae_model, body_model, batch_size, animation)
    smpl_forward(imu_flat, torch.squeeze(gt), smpl_vae_model, body_model, batch_size, animation)
    # Visualize the ground truth
    # pose = torch.reshape(gt, [batch_size, 72])
    # pose = torch.reshape(pose, [batch_size, 72])
    # batsz = pose.shape[0]
    # faces = body_model.faces
    # vts, jts = model_forward(body_model, pose, batsz)
    # # Visualize
    # body_from_vertices(vts, faces, animation)


def eval_vae(vae_ver=0, batch_size=64, batch_id=0, imu_start=0, imu_end=1):
    # load Convo_VAE model
    vae_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
                       'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    A_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
              'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)
    with open(vae_config_fname, 'r') as f_2:
        try:
            config_vae = yaml.safe_load(f_2)
        except yaml.YAMLError as exc:
            print(exc)
    # load VAE model
    vae_model = MyVAE(**config_vae['model_params'])
    conv_trained_fname = os.path.join(config_vae['logging_params']['save_dir'],
                                      config_vae['model_params']['name'], 'version_{}'.format(vae_ver),
                                      'checkpoints', 'last.ckpt')
    exp2 = VAEXperiment(vae_model, config_vae['exp_params'])
    vae_model = exp2.load_from_checkpoint(conv_trained_fname, vae_model=vae_model,
                                          params=config_vae['exp_params'])
    vae_model.eval()
    # test_loader = exp2.trainer.datamodule.test_dataloader()

    # test model with test dataset
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_len = len(list(iter(test_loader)))
    print('Number of batches: {}'.format(test_len))
    e = list(iter(test_loader))[batch_id]

    # e = next(iter(test_examples))
    imu, gt = e

    # test ConvoVAE model with reconstructed data
    A = torch.load(A_fname)
    m = exp2.h_in
    P_T = exp2.P_T
    imu_flat = torch.squeeze(imu)
    # (b, h_out) * (h_out, h_in) + (b, h_in) -> (b, h_in)
    y_batch = matmul_A(imu_flat, A)
    recons = vae_model(y_batch, A=A)[0]  # [b, h_out]
    dir_name = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/logs/VanillaVAE/version_{}'\
        .format(vae_ver)
    plot_reconstruction_data(recons.cpu().data.detach().numpy(), imu_flat.cpu().data.detach().numpy(),
                             imu_start, imu_end, dir_name)


def eval_smpl_vae(smpl_vae_ver=0, batch_id=0):
    fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/smplvae.yaml'
    with open(fname, 'r') as f_2:
        try:
            config = yaml.safe_load(f_2)
        except yaml.YAMLError as exc:
            print(exc)
    model = SMPLVAE(config['model_params']['in_channels'], config['model_params']['latent_dim'],
                    config['model_params']['h_in'], config['model_params']['h_out'])
    ckpt_fname = os.path.join(config['logging_params']['save_dir'],
                                      config['model_params']['name'], 'version_{}'.format(smpl_vae_ver),
                                      'checkpoints', 'last.ckpt')
    exp = SMPLexperiment(model, config['exp_params'])
    model = exp.load_from_checkpoint(ckpt_fname, vae_model=model, params=config['exp_params'])
    model.eval()
    # test model with test dataset
    batch_size = 60
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_len = len(list(iter(test_loader)))
    print('test_len: {}'.format(test_len))
    e = list(iter(test_loader))[batch_id]

    # e = next(iter(test_examples))
    imu, gt = e  # imu: (b, 1, 204), gt: [b, 1, 72]
    # load body model
    bm_fname = '/home/hinguyen/Data/smpl/models/smpl_male.pkl'
    body_model = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    print('Model: {}'.format(body_model))

    smpl_forward(torch.squeeze(imu), torch.squeeze(gt), model, body_model, batch_size, True)

    # Visualize the ground truth
    pose = torch.reshape(gt, [batch_size, 72])
    batsz = pose.shape[0]
    faces = body_model.faces
    vts, jts = model_forward(body_model, pose, batsz)
    # Visualize
    body_from_vertices(vts, faces, True)


def eval_baselines(vae_ver=0, batch_size=64, lasso_a=0.1, log_interval=10):
    vae_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
                       'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    dip_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
                       'logs/DIPVAE/version_{}/config.yaml'.format(vae_ver)
    A_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
              'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)
    saved_dir = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
              'logs/VanillaVAE/version_{}/results.npz'.format(vae_ver)
    with open(vae_config_fname, 'r') as f_2:
        try:
            config_vae = yaml.safe_load(f_2)
        except yaml.YAMLError as exc:
            print(exc)
    with open(dip_config_fname, 'r') as f_2:
        try:
            config_dip = yaml.safe_load(f_2)
        except yaml.YAMLError as exc:
            print(exc)
    # load VAE model
    vae_model = MyVAE(**config_vae['model_params'])
    vae_trained_fname = os.path.join(config_vae['logging_params']['save_dir'],
                                      config_vae['model_params']['name'], 'version_{}'.format(vae_ver),
                                      'checkpoints', 'last.ckpt')
    exp_vae = VAEXperiment(vae_model, config_vae['exp_params'])
    vae_model = exp_vae.load_from_checkpoint(vae_trained_fname, vae_model=vae_model, params=config_vae['exp_params'])
    vae_model.eval()
    # load DIP model
    dip_model = DIPVAE(**config_dip['model_params'])
    dip_trained_fname = os.path.join(config_dip['logging_params']['save_dir'],
                                      config_dip['model_params']['name'], 'version_{}'.format(vae_ver),
                                      'checkpoints', 'last.ckpt')
    exp_dip = DIPExperiment(dip_model, config_dip['exp_params'])
    dip_model = exp_dip.load_from_checkpoint(dip_trained_fname, vae_model=dip_model, params=config_dip['exp_params'])
    dip_model.eval()

    # test model with test dataset
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_len = len(test_dataset)
    print('Number of batches: {} -- version: {}'.format(test_len // batch_size, vae_ver))

    # VAE parameters
    A = torch.load(A_fname)
    m = exp_vae.h_in
    n = exp_vae.h_out
    vae_loss = []
    vae_time = []

    # Lasso parameters
    lasso_loss = []
    lasso_time = []
    lasso = Lasso(alpha=lasso_a, tol=1e-4)

    # DIP parameters
    dip_loss = []
    dip_time = []

    for batch_id, (imu, gt) in enumerate(test_loader):
        imu_flat = torch.squeeze(imu)
        # VAE
        noise = torch.normal(mean=0, std=exp_vae.noise_std, size=(batch_size, m))
        y_batch = matmul_A(imu_flat, A, noise)
        t0_vae = time.time()
        recons_vae = vae_model(y_batch, A=A)[0]  # [b, n]
        vae_time.append(round(time.time() - t0_vae, 2))
        loss_vae = get_l2_norm(recons_vae.cpu().detach().numpy(), imu_flat.cpu().detach().numpy())
        vae_loss.append(loss_vae)

        # Lasso
        y_lasso = y_batch.cpu().detach().numpy()
        t0_lasso = time.time()
        lasso.fit(X=A, y=y_lasso.T)
        lasso_time.append(round(time.time() - t0_lasso, 2))
        recons_lasso = lasso.coef_.reshape([batch_size, n])
        loss_lasso = get_l2_norm(recons_lasso, imu_flat.cpu().detach().numpy())
        lasso_loss.append(loss_lasso)

        # DIP
        y_dip = dip_model.get_input(imu_flat, exp_dip.dip_positions, exp_dip.noise_std, exp_dip.P_T, exp_dip.curr_device)
        t0_dip = time.time()
        recons_dip = dip_model(y_dip, positions=exp_dip.dip_positions)[0]
        dip_time.append(round(time.time() - t0_dip, 2))
        loss_dip = get_l2_norm(recons_dip.cpu().detach().numpy(), imu_flat.cpu().detach().numpy())
        dip_loss.append(loss_dip)

        if batch_id % log_interval == 0:
            print('Batch: %4d: Loss_VAE / Loss_Lasso / Loss_DIP: %5.3f / %5.3f / %5.3f '
                  '--- Time_VAE / Time_Lasso / Time_DIP: %5.3f / %5.3f / %5.3f'
                  % (batch_id, np.mean(vae_loss), np.mean(lasso_loss), np.mean(dip_loss),
                     np.mean(vae_time), np.mean(lasso_time), np.mean(dip_time)))

    np.savez(saved_dir, vae_loss=vae_loss, lasso_loss=lasso_loss, dip_loss=dip_loss,
             vae_time=vae_time, lasso_time=lasso_time, dip_time=dip_time)


def plot_results(vae_vers=[0], metric='mn'):
    matplotlib.use('TkAgg')
    vae_loss_arr = []
    vae_time_arr = []
    lasso_loss_arr = []
    lasso_time_arr = []
    dip_loss_arr = []
    dip_time_arr = []
    for v in vae_vers:
        f_name = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/' \
                    'logs/VanillaVAE/version_{}/results.npz'.format(v)
        results = np.load(f_name)
        lasso_loss_arr.append(results['lasso_loss'])
        lasso_time_arr.append(results['lasso_time'])
        vae_loss_arr.append(results['vae_loss'])
        vae_time_arr.append(results['vae_time'])
        dip_loss_arr.append(results['dip_loss'])
        dip_time_arr.append(results['dip_time'])

    if metric == 'mn':
        x = ['{}'.format(24*k) for k in range(1, len(vae_loss_arr) + 1)]
        plt.errorbar(x, np.mean(vae_loss_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='VAE')
        plt.errorbar(x, np.mean(lasso_loss_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='Lasso')
        plt.errorbar(x, np.mean(dip_loss_arr, axis=1), yerr=np.std(dip_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='DIP')
        plt.xlabel('Number of measurements (m)')
        plt.ylabel('Mean square error')
    elif metric == 'csnr':
        x = [k for k in range(0, 35, 5)]
        plt.errorbar(x, np.mean(vae_loss_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='VAE')
        plt.errorbar(x, np.mean(lasso_loss_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='Lasso')
        plt.errorbar(x, np.mean(dip_loss_arr, axis=1), yerr=np.std(dip_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='DIP')
        plt.xlabel('CSNR (dB)')
        plt.ylabel('Mean square error')
    else:
        x = ['{}'.format(24*k) for k in range(1, len(vae_loss_arr) + 1)]
        plt.errorbar(x, np.mean(vae_time_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='VAE')
        plt.errorbar(x, np.mean(lasso_time_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1),
                     fmt='-o', capsize=4, label='Lasso')
        plt.errorbar(x, np.mean(dip_time_arr, axis=1), yerr=np.std(dip_time_arr, axis=1),
                     fmt='-o', capsize=4, label='DIP')
        plt.xlabel('Number of measurements (m)')
        plt.ylabel('Decoding time (s)')

    plt.grid(linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # IMU map:
    imu_map = {
        'head:': 0,
        'spine2': 1,
        'belly': 2,
        'lchest': 3,
        'rchest': 4,
        'lshoulder': 5,
        'rshoulder': 6,
        'lelbow': 7,
        'relbow': 8,
        'lhip': 9,
        'rhip': 10,
        'lknee': 11,
        'rknee': 12,
        'lwrist': 13,
        'rwrist': 14,
        'lankle': 15,
        'rankle': 16
    }

    # eval_models(vae_ver=8, spml_vae_ver=24, batch_size=60, batch_id=666, animation=False)  # 8976
    # eval_vae(vae_ver=169, batch_size=60, batch_id=156, imu_start=imu_map['lwrist'], imu_end=imu_map['rwrist'])
    # eval_smpl_vae(smpl_vae_ver=24, batch_id=99)
    # eval_baselines(vae_ver=13, batch_size=60, lasso_a=0.0001, log_interval=100)
    plot_results(vae_vers=[k for k in range(0, 7)], metric='mn')