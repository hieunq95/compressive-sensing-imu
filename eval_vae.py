import os
import time
import math
import pickle as pkl
import yaml
import smplx
import pyrender
import trimesh
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from torch.utils.data import DataLoader
from torch.nn import functional as F
from train_smpl_vae import SMPLexperiment
from vae import SMPLVAE, MyVAE, DIPVAE
from train_mlp_vae import VAEXperiment
from train_dip import DIPExperiment
from dataset import IMUDataset
from imu_utils import matmul_A, plot_reconstruction_data, get_mean_loss, get_imu_positions, rot_matrix_to_aa
from sklearn.linear_model import Lasso

# matplotlib parameters
font_size = 18
legend_size = 14
xtick_size = 14
ytick_size = 14
line_width = 2
back_end ='TkAgg'

file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
bm_fname = '/home/hinguyen/Data/smpl/models/smpl_male.pkl'

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


def body_from_vertices(vertices, faces, key_frame=False, animation=False, color=None):
    if color == 'gt':
        mesh_color = [255.0 / 255, 51.0 / 255, 51.0 / 255]
    elif color == 'vae':
        mesh_color = [224.0 / 255, 224.0 / 255, 225.0 / 255]
    elif color == 'lasso':
        mesh_color = [51.0 / 255, 153.0 / 255, 255.0 / 255]
    elif color == 'lasso-opt':
        mesh_color = [102.0 / 255, 255.0 / 255, 102.0 / 255]
    elif color == 'dip':
        mesh_color = [255.0 / 255, 153.0 / 255, 255.0 / 255]
    else:
        if key_frame:
            mesh_color = [255.0 / 255, 51.0 / 255, 51.0 / 255]
        else:
            mesh_color = [224.0 / 255, 224.0 / 255, 225.0 / 255]

    seq_len = vertices.shape[0]
    scene = pyrender.Scene()
    mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces, vertex_colors=[mesh_color] * len(vertices[0]))
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node, name='mesh')

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
            mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces)
            mesh_node = pyrender.Mesh.from_trimesh(mesh)

            viewer.render_lock.acquire()
            scene = clear_scene(scene)
            scene.add(mesh_node, name='mesh')
            viewer.render_lock.release()
            t += 1
        else:
            mesh = trimesh.Trimesh(vertices=vertices[i], faces=faces, vertex_colors=[mesh_color] * len(vertices[i]))
            mesh_node = pyrender.Mesh.from_trimesh(mesh)

            viewer.render_lock.acquire()
            scene = clear_scene(scene)
            scene.add(mesh_node, name='mesh')
            viewer.render_lock.release()


def smpl_forward(imu, gt, vae_model, body_model, key_frame=False, animation=False, color=None):
    pose = vae_model(imu, labels=gt)[0]  # [b, 1, 72]

    faces = body_model.faces
    batsz = pose.shape[0]
    vts, jts = model_forward(body_model, pose, batsz)
    # Visualize
    body_from_vertices(vts, faces, key_frame, animation, color)


def reconstruct_pose(vae_ver=0, smpl_vae_ver=0, batch_size=60, batch_id=0, l1_penalty=1e-5):
    vae_config_fname = 'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    dip_config_fname = 'logs/DIPVAE/version_{}/config.yaml'.format(vae_ver)
    smpl_vae_config_fname = 'configs/smplvae.yaml'
    A_fname = 'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)
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
    with open(smpl_vae_config_fname, 'r') as f_1:
        try:
            config_smpl_vae = yaml.safe_load(f_1)
        except yaml.YAMLError as exc:
            print(exc)

        # load body model
    body_model = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    print('Model: {}'.format(body_model))

    # load SMPL_VAE model
    smpl_vae_model = SMPLVAE(**config_smpl_vae['model_params'])
    smpl_trained_fname = os.path.join(config_smpl_vae['logging_params']['save_dir'],
                                      config_smpl_vae['model_params']['name'], 'version_{}'.format(smpl_vae_ver),
                                      'checkpoints', 'last.ckpt')

    smpl_exp = SMPLexperiment(smpl_vae_model, config_smpl_vae['exp_params'])
    smpl_vae_model = smpl_exp.load_from_checkpoint(smpl_trained_fname, vae_model=smpl_vae_model,
                                                   params=config_smpl_vae['exp_params'])
    smpl_vae_model.eval()

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
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_len = len(test_dataset)
    print('Number of batches: {} -- version: {}'.format(test_len // batch_size, vae_ver))

    # VAE parameters
    m = exp_vae.h_in
    n = exp_vae.h_out
    A = torch.load(A_fname)
    A_lasso_opt = torch.normal(mean=0, std=1/math.sqrt(m), size=[m, n])

    # Lasso parameter
    lasso_pow = Lasso(alpha=l1_penalty, tol=1e-3)
    lasso_opt = Lasso(alpha=l1_penalty, tol=1e-3)

    (imu, gt) = list(iter(test_loader))[batch_id]
    noise = torch.normal(mean=0, std=exp_vae.noise_std, size=[batch_size, m])
    y_batch = matmul_A(torch.squeeze(imu), A, noise)
    y_dip = dip_model.get_input(torch.squeeze(imu), exp_dip.dip_positions, exp_dip.noise_std, exp_dip.P_T, imu.device)
    y_lasso_opt = matmul_A(torch.squeeze(imu), A_lasso_opt, noise)

    # Reconstructed data
    recons_vae = vae_model(y_batch, A=A)[0]
    recons_dip = dip_model(y_dip, positions=exp_dip.dip_positions)[0]
    lasso_pow.fit(X=A, y=y_batch.cpu().detach().numpy().T)
    lasso_opt.fit(X=A_lasso_opt, y=y_lasso_opt.cpu().detach().numpy().T)
    recons_lasso_pow = lasso_pow.coef_.reshape([batch_size, n])
    recons_lasso_pow = torch.Tensor(recons_lasso_pow)
    recons_lasso_opt = lasso_opt.coef_.reshape([batch_size, n])
    recons_lasso_opt = torch.Tensor(recons_lasso_opt)

    # Ground truth pose
    smpl_forward(torch.squeeze(imu), torch.squeeze(gt), smpl_vae_model, body_model, False, False, 'gt')
    # Reconstructed pose
    smpl_forward(recons_vae, torch.squeeze(gt), smpl_vae_model, body_model, False, False, 'vae')
    smpl_forward(recons_lasso_pow, torch.squeeze(gt), smpl_vae_model, body_model, False, False, 'lasso')
    smpl_forward(recons_lasso_opt, torch.squeeze(gt), smpl_vae_model, body_model, False, False, 'lasso-opt')
    smpl_forward(recons_dip, torch.squeeze(gt), smpl_vae_model, body_model, False, False, 'dip')


def latent_interpolation(vae_ver=0, spml_vae_ver=0, batch_size=60, batch_start=0, batch_end=1):
    matplotlib.use(back_end)
    smpl_vae_config_fname = 'configs/smplvae.yaml'
    vae_config_fname = 'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    A_fname = 'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)

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
    body_model = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    print('Model: {}'.format(body_model))

    # load SMPL_VAE model
    smpl_vae_model = SMPLVAE(**config_smpl_vae['model_params'])
    smpl_trained_fname = os.path.join(config_smpl_vae['logging_params']['save_dir'],
                                      config_smpl_vae['model_params']['name'], 'version_{}'.format(spml_vae_ver),
                                      'checkpoints', 'last.ckpt')

    smpl_exp = SMPLexperiment(smpl_vae_model, config_smpl_vae['exp_params'])
    smpl_vae_model = smpl_exp.load_from_checkpoint(smpl_trained_fname, vae_model=smpl_vae_model,
                                                   params=config_smpl_vae['exp_params'])
    smpl_vae_model.eval()

    # load VAE model
    vae_model = MyVAE(**config_vae['model_params'])
    vae_trained_fname = os.path.join(config_vae['logging_params']['save_dir'],
                                      config_vae['model_params']['name'], 'version_{}'.format(vae_ver),
                                      'checkpoints', 'last.ckpt')
    vae_exp = VAEXperiment(vae_model, config_vae['exp_params'])
    vae_model = vae_exp.load_from_checkpoint(vae_trained_fname, vae_model=vae_model, params=config_vae['exp_params'])
    vae_model.eval()

    # test models with test dataset
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    num_batches = len(list(iter(test_loader)))
    print('Number of batches: {}'.format(num_batches))
    frame_start = list(iter(test_loader))[batch_start]
    frame_end = list(iter(test_loader))[batch_end]
    (imu_start, gt_start) = frame_start
    (imu_end, gt_end) = frame_end

    # test VAE model with reconstructed data
    A = torch.load(A_fname)
    m = vae_exp.h_in
    # first key frame
    imu_flat_start = torch.squeeze(imu_start)
    noise = torch.normal(mean=0, std=vae_exp.noise_std, size=[batch_size, m])
    y_batch_start = matmul_A(imu_flat_start, A, noise)
    # second key frame
    imu_flat_end = torch.squeeze(imu_end)
    y_batch_end = matmul_A(imu_flat_end, A, noise)
    # forward and observe key frames
    z_1 = vae_model.model.encode(y_batch_start)
    z_1 = vae_model.model.reparameterize(z_1[0], z_1[1])
    z_2 = vae_model.model.encode(y_batch_end)
    z_2 = vae_model.model.reparameterize(z_2[0], z_2[1])
    smpl_forward(vae_exp.model.decode(z_1), torch.squeeze(gt_start), smpl_vae_model, body_model, True)
    smpl_forward(vae_exp.model.decode(z_2), torch.squeeze(gt_end), smpl_vae_model, body_model, True)
    # interpolation
    z_range = np.linspace(0.0, 1.0, num=10)
    for alpha in z_range:
        z = alpha * z_2 + (1 - alpha) * z_1
        smpl_forward(vae_exp.model.decode(z), torch.squeeze(gt_start), smpl_vae_model, body_model, False)

    # Visualize the ground truth
    # pose = torch.reshape(gt_start, [batch_size, 72])
    # pose = torch.reshape(pose, [batch_size, 72])
    # batsz = pose.shape[0]
    # faces = body_model.faces
    # vts, jts = model_forward(body_model, pose, batsz)
    # # Visualize
    # body_from_vertices(vts, faces, animation)


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
    test_dataset = IMUDataset(file_path, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_len = len(list(iter(test_loader)))
    print('test_len: {}'.format(test_len))
    e = list(iter(test_loader))[batch_id]

    # e = next(iter(test_examples))
    imu, gt = e  # imu: (b, 1, 204), gt: [b, 1, 72]
    # load body model
    body_model = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    print('Model: {}'.format(body_model))

    smpl_forward(torch.squeeze(imu), torch.squeeze(gt), model, body_model, False, True)

    # Visualize the ground truth
    pose = torch.reshape(gt, [batch_size, 72])
    batsz = pose.shape[0]
    faces = body_model.faces
    vts, jts = model_forward(body_model, pose, batsz)
    # Visualize
    body_from_vertices(vts, faces, False, True)


def eval_decoding_time(vae_ver=0, batch_sizes=[60], l1_penalty=0.0001, log_interval=100, plot=False):
    vae_config_fname = 'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    dip_config_fname = 'logs/DIPVAE/version_{}/config.yaml'.format(vae_ver)
    A_fname = 'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)
    saved_dir = 'logs/DecodingTime'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
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

    # Plot if it is required
    if plot:
        matplotlib.use(back_end)
        lasso_pow_time_arr = []
        lasso_opt_time_arr = []
        vae_time_arr = []
        dip_time_arr = []
        for batch_size in batch_sizes:
            f_name = os.path.join(saved_dir, 'btz_size_{}.npz'.format(batch_size))
            results = np.load(f_name)
            lasso_pow_time_arr.append(results['lasso_pow_time'])
            lasso_opt_time_arr.append(results['lasso_opt_time'])
            vae_time_arr.append(results['vae_time'])
            dip_time_arr.append(results['dip_time'])
            # print(results['vae_time'])
        # x = ['60', '120', '180', '240', '300', '360', '420']  # m = 168
        x = ['10', '20', '30', '40', '50', '60', '70']  # rounded values

        # print('vae_time_arr: {}'.format(vae_time_arr))
        y_vae, y_lasso_pow, y_lasso_opt, y_dip = [], [], [], []
        y_vae_err, y_lasso_pow_err, y_lasso_opt_err, y_dip_err = [], [], [], []
        for i in range(len(vae_time_arr)):
            y_vae.append(np.mean(vae_time_arr[i]))
            y_vae_err.append(np.std(vae_time_arr[i]/2))
            y_lasso_pow.append(np.mean(lasso_pow_time_arr[i]))
            y_lasso_pow_err.append(np.std(lasso_pow_time_arr[i]/2))
            y_lasso_opt.append(np.mean(lasso_opt_time_arr[i]))
            y_lasso_opt_err.append(np.std(lasso_opt_time_arr[i]/2))
            y_dip.append(np.mean(dip_time_arr[i]))
            y_dip_err.append(np.std(dip_time_arr[i]/2))
        plt.errorbar(x, y_vae, yerr=y_vae_err, fmt='-o', capsize=4, linewidth=line_width, label='CS-VAE')
        plt.errorbar(x, y_lasso_pow, yerr=y_lasso_pow_err, fmt='--o', capsize=4, linewidth=line_width, label='Lasso')
        plt.errorbar(x, y_lasso_opt, yerr=y_lasso_opt_err, fmt='--o', capsize=4, linewidth=line_width, label=r'Lasso w.o.$P_T$')
        plt.errorbar(x, y_dip, yerr=y_dip_err, fmt='--o', capsize=4, linewidth=line_width, label='DIP')
        plt.xlabel('Number of input samples ($10^3$)', fontsize=font_size)
        plt.ylabel('Decoding time (s)', fontsize=font_size)
        plt.yscale('log')
        plt.subplots_adjust(left=0.16,
                            bottom=0.15,
                            right=0.98,
                            top=0.95,
                            wspace=0.2,
                            hspace=0.255)

        plt.grid(linestyle='--')
        plt.xticks(fontsize=xtick_size)
        plt.yticks(fontsize=ytick_size)
        legend = plt.legend(fontsize=legend_size, loc='best', ncol=2)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 0, 0))
        plt.show()
        return

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

    # VAE parameters
    A = torch.load(A_fname)
    m = exp_vae.h_in
    n = exp_vae.h_out

    for batch_id in batch_sizes:
        vae_time = []
        # Lasso parameters
        lasso_pow_time = []
        lasso_opt_time = []
        lasso_pow = Lasso(alpha=l1_penalty, tol=1e-3)
        lasso_opt = Lasso(alpha=l1_penalty, tol=1e-3)
        A_opt = torch.normal(mean=0, std=1 / math.sqrt(m), size=[m, n])
        # DIP parameters
        dip_time = []
        # test model with test dataset
        test_dataset = IMUDataset(file_path, mode='test', transform=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_id, shuffle=False, drop_last=True)
        for batch_id, (imu, gt) in enumerate(test_loader):
            if batch_id == 0:
                print('\nBatch size: {}, running on: {}'.format(batch_id, imu.device))
            imu_flat = torch.squeeze(imu)
            # VAE
            noise = torch.normal(mean=0, std=exp_vae.noise_std, size=(batch_id, m))
            y_batch = matmul_A(imu_flat, A, noise)
            t0_vae = time.time()
            recons_vae = vae_model(y_batch, A=A)[0]  # [b, n]
            vae_time.append(time.time() - t0_vae)  # vae_time: (nb, )

            # Lasso
            y_lasso_pow = y_batch.cpu().detach().numpy()
            t0_lasso = time.time()
            lasso_pow.fit(X=A, y=y_lasso_pow.T)
            lasso_pow_time.append(time.time() - t0_lasso)
            recons_lasso = lasso_pow.coef_.reshape([batch_id, n])
            y_lasso_opt = matmul_A(imu_flat, A_opt, noise)
            t1_lasso = time.time()
            lasso_opt.fit(X=A_opt, y=y_lasso_opt.cpu().detach().numpy().T)
            lasso_opt_time.append(time.time() - t1_lasso)
            recons_lasso_opt = lasso_opt.coef_.reshape([batch_id, n])

            # DIP
            y_dip = dip_model.get_input(imu_flat, exp_dip.dip_positions, exp_dip.noise_std, exp_dip.P_T,
                                        exp_dip.curr_device)
            t0_dip = time.time()
            recons_dip = dip_model(y_dip, positions=exp_dip.dip_positions)[0]
            dip_time.append(time.time() - t0_dip)

            if batch_id % log_interval == 0:
                print(
                    'Batch: %4d: Time_VAE / Time_Lasso_pow / Time_Lasso_opt / Time_DIP: %5.3f / %5.3f / %5.3f / %5.3f'
                    % (batch_id, np.mean(vae_time), np.mean(lasso_pow_time), np.mean(lasso_opt_time), np.mean(dip_time))
                      )

        saved_file = os.path.join(saved_dir, 'btz_size_{}.npz'.format(batch_id))
        np.savez(saved_file, vae_time=vae_time, lasso_pow_time=lasso_pow_time,
                 lasso_opt_time=lasso_opt_time, dip_time=dip_time)


def eval_baselines(vae_ver=0, batch_size=60, l1_penalty=0.0001, log_interval=10):
    """
    Use to evaluate the baselines with metric is either 'mn' or 'noise'. With metric 'time', use 'eval_decoding_time'
    instead
    """
    vae_config_fname = 'logs/VanillaVAE/version_{}/config.yaml'.format(vae_ver)
    dip_config_fname = 'logs/DIPVAE/version_{}/config.yaml'.format(vae_ver)
    A_fname = 'logs/VanillaVAE/version_{}/A.pt'.format(vae_ver)
    saved_dir = 'logs/VanillaVAE/version_{}/results.npz'.format(vae_ver)
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
    lasso_pow_loss = []
    lasso_pow_time = []
    lasso_pow = Lasso(alpha=l1_penalty, tol=1e-3)
    A_opt = torch.normal(mean=0, std=1/math.sqrt(m), size=[m, n])
    lasso_opt_loss = []
    lasso_opt_time = []
    lasso_opt = Lasso(alpha=l1_penalty, tol=1e-3)

    # DIP parameters
    dip_loss = []
    dip_time = []

    for batch_id, (imu, gt) in enumerate(test_loader):
        if batch_id == 0:
            print('Running on: {}'.format(imu.device))
        imu_flat = torch.squeeze(imu)
        # VAE
        noise = torch.normal(mean=0, std=exp_vae.noise_std, size=(batch_size, m))
        y_batch = matmul_A(imu_flat, A, noise)
        t0_vae = time.time()
        recons_vae = vae_model(y_batch, A=A)[0]  # [b, n]
        vae_time.append(time.time() - t0_vae)
        loss_vae = get_mean_loss(recons_vae.cpu().detach().numpy(), imu_flat.cpu().detach().numpy())
        vae_loss.append(loss_vae)

        # We evaluate two scenarios with Lasso, i.e., with and without power constrain
        # by using different measurement matrices
        # scenario 1: With pow constraint
        y_lasso_pow = matmul_A(imu_flat, A, noise)
        t0_lasso = time.time()
        lasso_pow.fit(X=A, y=y_lasso_pow.T)
        lasso_pow_time.append(time.time() - t0_lasso)
        recons_lasso_pow = lasso_pow.coef_.reshape([batch_size, n])
        loss = get_mean_loss(recons_lasso_pow, imu_flat.cpu().detach().numpy())
        lasso_pow_loss.append(loss)
        # scenario 2: Without pow constraint
        y_lasso_opt = matmul_A(imu_flat, A_opt, noise)
        t1_lasso = time.time()
        lasso_opt.fit(X=A_opt, y=y_lasso_opt.T)
        lasso_opt_time.append(time.time() - t1_lasso)
        recons_lasso_opt = lasso_opt.coef_.reshape([batch_size, n])
        loss = get_mean_loss(recons_lasso_opt, imu_flat.cpu().detach().numpy())
        lasso_opt_loss.append(loss)

        # DIP
        y_dip = dip_model.get_input(imu_flat, exp_dip.dip_positions, exp_dip.noise_std, exp_dip.P_T, exp_dip.curr_device)
        t0_dip = time.time()
        recons_dip = dip_model(y_dip, positions=exp_dip.dip_positions)[0]
        dip_time.append(time.time() - t0_dip)
        loss_dip = get_mean_loss(recons_dip.cpu().detach().numpy(), imu_flat.cpu().detach().numpy())
        dip_loss.append(loss_dip)

        if batch_id % log_interval == 0:
            print('Batch: %4d: Loss_VAE / Loss_Lasso_pow / Loss_Lasso_opt / Loss_DIP: %5.3f / %5.3f / %5.3f / %5.3f '
                  '--- Time_VAE / Time_Lasso_pow / Time_Lasso_opt / Time_DIP: %5.3f / %5.3f / %5.3f / %5.3f'
                  % (batch_id, np.mean(vae_loss), np.mean(lasso_pow_loss), np.mean(lasso_opt_loss), np.mean(dip_loss),
                     np.mean(vae_time), np.mean(lasso_pow_time), np.mean(lasso_opt_time), np.mean(dip_time)))

    np.savez(saved_dir, vae_loss=vae_loss, lasso_pow_loss=lasso_pow_loss, lasso_opt_loss=lasso_opt_loss, dip_loss=dip_loss,
             vae_time=vae_time, lasso_pow_time=lasso_pow_time, lasso_opt_time=lasso_opt_time, dip_time=dip_time)


def plot_results(vae_vers=[0], metric='mn'):
    matplotlib.use(back_end)
    vae_loss_arr = []
    vae_time_arr = []
    lasso_pow_loss = []
    lasso_pow_time_arr = []
    lasso_opt_loss = []
    lasso_opt_time_arr = []
    dip_loss_arr = []
    dip_time_arr = []
    for v in vae_vers:
        f_name = 'logs/VanillaVAE/version_{}/results.npz'.format(v)
        results = np.load(f_name)
        lasso_pow_loss.append(results['lasso_pow_loss'])
        lasso_pow_time_arr.append(results['lasso_pow_time'])
        lasso_opt_loss.append(results['lasso_opt_loss'])
        lasso_opt_time_arr.append(results['lasso_opt_time'])
        vae_loss_arr.append(results['vae_loss'])
        vae_time_arr.append(results['vae_time'])
        dip_loss_arr.append(results['dip_loss'])
        dip_time_arr.append(results['dip_time'])

    if metric == 'mn':
        # We just plot the error bars with half of the std values for clearer illustration
        x = ['48', '72', '120', '144', '168', '192']
        plt.errorbar(x, np.mean(vae_loss_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1) / 2,
                     fmt='-o', capsize=4, linewidth=line_width, label='CS-VAE')
        plt.errorbar(x, np.mean(lasso_pow_loss, axis=1), yerr=np.std(lasso_pow_loss, axis=1) / 2,
                     fmt='--o', capsize=4, linewidth=line_width, label='Lasso')
        plt.errorbar(x, np.mean(lasso_opt_loss, axis=1), yerr=np.std(lasso_opt_loss, axis=1) / 2,
                     fmt='--o', capsize=4, linewidth=line_width, label=r'Lasso w.o.$P_T$')
        plt.errorbar(x, np.mean(dip_loss_arr, axis=1), yerr=np.std(dip_loss_arr, axis=1) / 2,
                     fmt='--o', capsize=4, linewidth=line_width, label='DIP')
        plt.xlabel('Number of measurements', fontsize=font_size)
        plt.ylabel('Mean square error', fontsize=font_size)
    elif metric == 'noise':
        x = ['1', '5', '10', '50', '100', '500']
        plt.errorbar(x, np.mean(vae_loss_arr, axis=1), yerr=np.std(vae_loss_arr, axis=1) / 2,
                     fmt='-o', capsize=4, linewidth=line_width, label='CS-VAE')
        plt.errorbar(x, np.mean(lasso_pow_loss, axis=1), yerr=np.std(lasso_pow_loss, axis=1) / 2,
                     fmt='--o', capsize=4, linewidth=line_width, label='Lasso')
        plt.errorbar(x, np.mean(lasso_opt_loss, axis=1), yerr=np.std(lasso_opt_loss, axis=1) / 2,
                     fmt='--o', capsize=4, linewidth=line_width, label=r'Lasso w.o.$P_T$')
        plt.errorbar(x, np.mean(dip_loss_arr, axis=1), yerr=np.std(dip_loss_arr, axis=1) / 2,
                     fmt='--o', capsize=4, linewidth=line_width, label='DIP')
        plt.xlabel(r'$\sigma_N\ (10^{-4})$', fontsize=font_size)
        plt.ylabel('Mean square error', fontsize=font_size)
    else:
        x = ['24', '48', '72', '120', '144', '168', '192']
        plt.errorbar(x, np.mean(vae_time_arr, axis=1), yerr=np.std(vae_time_arr, axis=1),
                     fmt='-o', capsize=4, linewidth=line_width, label='CS-VAE')
        plt.errorbar(x, np.mean(lasso_pow_time_arr, axis=1), yerr=np.std(lasso_pow_time_arr, axis=1),
                     fmt='-o', capsize=4, linewidth=line_width, label='Lasso')
        plt.errorbar(x, np.mean(lasso_opt_time_arr, axis=1), yerr=np.std(lasso_opt_time_arr, axis=1),
                     fmt='-o', capsize=4, linewidth=line_width, label=r'Lasso w.o.$P_T$')
        plt.errorbar(x, np.mean(dip_time_arr, axis=1), yerr=np.std(dip_time_arr, axis=1),
                     fmt='-o', capsize=4, linewidth=line_width, label='DIP')
        plt.xlabel('Number of measurements', fontsize=font_size)
        plt.ylabel('Decoding time (s)', fontsize=font_size)

    plt.subplots_adjust(left=0.16,
                        bottom=0.15,
                        right=0.98,
                        top=0.95,
                        wspace=0.2,
                        hspace=0.255)

    plt.grid(linestyle='--')
    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    legend = plt.legend(fontsize=legend_size, loc='best', ncol=1)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    plt.show()


def plot_imu_readings(subject='s_03/05', sensor='imu_ori', start_time=0, end_time=4, imu_position=8):
    matplotlib.use(back_end)
    pkl_path = file_path + 'DIP_IMU/{}.pkl'.format(subject)
    data = pkl.load(open(pkl_path, 'rb'), encoding='latin1')[sensor]
    keys = pkl.load(open(pkl_path, 'rb'), encoding='latin1').keys()
    print(keys)
    print('data.shape: {}'.format(data.shape))

    seq_len = data.shape[0]
    sampling_rate = 60
    if end_time >= int(seq_len / sampling_rate):
        end_time = int(seq_len / sampling_rate)

    t = np.arange(0, seq_len / sampling_rate, 1.0 / sampling_rate)
    if sensor == 'imu_acc':
        y = data[:, imu_position, :]
    else:
        y = rot_matrix_to_aa(np.reshape(data, [seq_len, 17 * 9]))
    print('t: {}'.format(t.shape))
    print('y: {}'.format(y.shape))
    t_display = t[start_time * sampling_rate: end_time * sampling_rate]
    y_display = y[start_time * sampling_rate: end_time * sampling_rate]
    print('y_display: {}'.format(y_display.shape))
    plt.subplots(figsize=(6, 2))
    if sensor == 'imu_acc':
        plt.plot(t_display, y_display[:, 0], label='X', linewidth=line_width)
        plt.plot(t_display, y_display[:, 1], label='Y', linewidth=line_width)
        plt.plot(t_display, y_display[:, 2], label='Z', linewidth=line_width)
        plt.ylabel('Acceleration', fontsize=font_size)
    else:
        plt.plot(t_display, y_display[:, imu_position], label='X', linewidth=line_width)
        plt.plot(t_display, y_display[:, imu_position+1], label='Y', linewidth=line_width)
        plt.plot(t_display, y_display[:, imu_position+2], label='Z', linewidth=line_width)
        plt.ylabel('Orientation', fontsize=font_size)
    plt.legend()
    # plt.grid(linestyle='--')
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    # plt.title('IMU ID: {}'.format(imu_position))
    plt.subplots_adjust(left=0.16,
                        bottom=0.3,
                        right=0.97,
                        top=0.95,
                        wspace=0.2,
                        hspace=0.255)
    plt.show()

    # Plot FFT of IMU data
    if sensor == 'imu_acc':
        fig, axs = plt.subplots(2)

        axs[0].plot(t_display, y_display[:, 0], label='X', linewidth=line_width)
        axs[0].plot(t_display, y_display[:, 1], label='Y', linewidth=line_width)
        axs[0].plot(t_display, y_display[:, 2], label='Z', linewidth=line_width)
        axs[0].set_ylabel('Acceleration', fontsize=font_size)
        axs[0].set_xlabel('Time (s)', fontsize=font_size)
        axs[0].legend()

        y_len = len(y_display[:, 0])
        t_step = 1.0 / sampling_rate

        yf_1 = rfft(y_display[:, 0])  # FFT for real-valued inputs
        xf_1 = rfftfreq(y_len, t_step)

        yf_2 = rfft(y_display[:, 1])
        xf_2 = rfftfreq(y_len, t_step)

        yf_3 = rfft(y_display[:, 2])
        xf_3 = rfftfreq(y_len, t_step)

        axs[1].stem(xf_1, np.abs(yf_1), label='X', linefmt='#1f77b4', markerfmt='o')
        # axs[1].stem(xf_2, np.abs(yf_2), label='Y', linefmt='#ff7f0e', markerfmt='*')
        # axs[1].stem(xf_3, 1.0 / y_len * np.abs(yf_3), label='Z', linefmt='#2ca02c', markerfmt='D')
        axs[1].set_ylabel('Amplitude', fontsize=font_size)
        axs[1].set_xlabel('Frequency (Hz)', fontsize=font_size)
        axs[1].legend()

        plt.subplots_adjust(left=0.16,
                            bottom=0.13,
                            right=0.97,
                            top=0.97,
                            wspace=0.2,
                            hspace=0.37)

        axs[0].tick_params(axis='both', which='major', labelsize=xtick_size)
        axs[1].tick_params(axis='both', which='major', labelsize=xtick_size)

        plt.legend()
        plt.show()


def visualize_pose(subject='s_03/05'):
    pkl_path = file_path + 'DIP_IMU/{}.pkl'.format(subject)
    data = pkl.load(open(pkl_path, 'rb'), encoding='latin1')
    # 'gt' (ground truth) has 23 joints -> (23+1)*3 pose parameters
    output = data['gt']
    print(data.keys())
    print('output.shape: {}'.format(output.shape))

    bm = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    faces = bm.faces
    print('Model: {}'.format(bm))

    # get pose and joint parameters from the IMU data, shape (seq_length, 17, 3, 3) -> (127,3)?
    pose = torch.from_numpy(output.reshape(-1, 72))
    print('pose.shape: {}'.format(pose.shape))
    batsz = pose.shape[0]
    vts, jts = model_forward(bm, pose, batsz)
    print('jts.shape: {}'.format(jts.shape))
    # Visualize
    body_from_vertices(vts, faces, False, True)
    # body_from_joints(joints=jts)


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
    EVALUATION = False
    # /--- ***** Part 1: Evaluate baseline algorithms before plotting results *******-------/
    if EVALUATION:
        # eval_baselines(vae_ver=1, batch_size=60, l1_penalty=1e-5, log_interval=10)  # version 0 to 13
        for i in range(0, 14):
            eval_baselines(vae_ver=i, batch_size=60, l1_penalty=1e-5, log_interval=10)  # version 0 to 13
        eval_decoding_time(vae_ver=5, batch_sizes=[60, 120, 180, 240, 300, 360, 420], plot=False)

    # /--- ***** Part 2: Get simulation results after evaluation *******-------/
    else:
        # Visualize IMU data and pretrained SMPL-VAE model
        plot_imu_readings(subject='s_01/04', sensor='imu_acc', start_time=0, end_time=8, imu_position=imu_map['lwrist'])
        visualize_pose(subject='s_01/04')

        eval_smpl_vae(smpl_vae_ver=24, batch_id=99)

        # Plot all results
        reconstruct_pose(vae_ver=5, smpl_vae_ver=24, batch_size=6, batch_id=4440)  # batch_id = 890,  1111, 4440

        # For interpolation, we need to train another VAE with kld_weight=0.0001 and h_in=168 for smoother transitions
        # between the key poses, but the reconstructed signals can be less accurate.
        latent_interpolation(vae_ver=14, spml_vae_ver=24, batch_size=60, batch_start=223, batch_end=447)

        # Plot line graphs
        plot_results(vae_vers=[1, 2, 3, 4, 5, 6], metric='mn')
        plot_results(vae_vers=[13, 12, 11, 10, 9, 8], metric='noise')
        eval_decoding_time(vae_ver=5, batch_sizes=[60, 120, 180, 240, 300, 360, 420], plot=True)