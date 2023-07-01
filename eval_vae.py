import os
import yaml
import smplx
import pyrender
import trimesh
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from train_smpl_vae import SMPLexperiment
from vae import SMPLVAE, MyVAE
from train_mlp_vae import VAEXperiment
from dataset import IMUDataset
from imu_utils import matmul_A, denormalize_acceleration, plot_reconstruction_data


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
    torch.manual_seed(1234)
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

    # test ConvoVAE model with reconstructed data
    A = torch.load(A_fname)
    print('A: {}'.format(A))
    m = exp2.h_in
    noise = exp2.noise_std * torch.randn((batch_size, m))  # m compressed measurements
    imu_flat = torch.squeeze(imu)
    y_batch = matmul_A(imu_flat, A, noise)
    # recons = conv_vae_model.generate(y_batch, A=A)  # [b, 1, h_out, tw]
    recons = vae_model(y_batch, A=A)[0]  # [b, h_out]

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
    noise = exp2.noise_std * torch.randn((batch_size, m))  # m compressed measurements
    imu_flat = torch.squeeze(imu)
    # (b, h_out) * (h_out, h_in) + (b, h_in) -> (b, h_in)
    y_batch = matmul_A(imu_flat, A, noise)
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

    eval_models(vae_ver=168, spml_vae_ver=24, batch_size=60, batch_id=126, animation=False)  # 8976
    # eval_vae(vae_ver=164, batch_size=60, batch_id=156, imu_start=imu_map['lwrist'], imu_end=imu_map['rwrist'])
    # eval_smpl_vae(smpl_vae_ver=24, batch_id=99)