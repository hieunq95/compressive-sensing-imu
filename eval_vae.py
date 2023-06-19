import os
import yaml
import smplx
import pyrender
import trimesh
import numpy as np
import torch
torch.manual_seed(1234)
from torch.utils.data import DataLoader
from torch.nn import functional as F
from smpl_vae.smpl_vae import SMPLVAE
from train_smpl_vae import SMPLexperiment
from vae import ConvoVAE
from train_vae import VAEXperiment
from dataset import IMUDataset


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

            i += 0.01
        else:
            tri_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            viewer.render_lock.acquire()
            scene = clear_scene(scene)
            scene.add(mesh, name='mesh')
            viewer.render_lock.release()

            i += 0.01


def smpl_forward(imu, gt, vae_model, body_model, batch_size, time_window, animation=False):
    y_hat = vae_model(imu, labels=gt)[0]  # [batch_size, 1, 72, tw]
    pose = np.transpose(y_hat.data.numpy(), (0, 3, 2, 1))  # [batch_size, tw, 72, 1]
    pose = torch.from_numpy(pose)
    pose = torch.reshape(pose, [batch_size * time_window, 72])
    faces = body_model.faces
    batsz = pose.shape[0]
    vts, jts = model_forward(body_model, pose, batsz)
    # Visualize
    body_from_vertices(vts, faces, animation)


def eval_model(conv_vae_ver=0, spml_vae_ver=0, ground_truth=False, animation=False):
    smpl_vae_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/smplvae.yaml'
    conv_vae_config_fname = '/home/hinguyen/Data/PycharmProjects/compressive-sensing-imu/configs/convvae.yaml'

    with open(smpl_vae_config_fname, 'r') as f_1:
        try:
            config_smpl_vae = yaml.safe_load(f_1)
        except yaml.YAMLError as exc:
            print(exc)

    with open(conv_vae_config_fname, 'r') as f_2:
        try:
            config_conv_vae = yaml.safe_load(f_2)
        except yaml.YAMLError as exc:
            print(exc)

    # load body model
    bm_fname = '/home/hinguyen/Data/smpl/models/smpl_male.pkl'
    body_model = smplx.create(model_path=bm_fname, model_type='smpl', gender='male', dtype=torch.float64)
    print('Model: {}'.format(body_model))

    # load SMPL_VAE model
    vae_models = {'SMPLVAE': SMPLVAE, 'ConvoVAE': ConvoVAE}
    smpl_vae_model = vae_models[config_smpl_vae['model_params']['name']](**config_smpl_vae['model_params'])
    smpl_trained_fname = os.path.join(config_smpl_vae['logging_params']['save_dir'],
                                      config_smpl_vae['model_params']['name'], 'version_{}'.format(spml_vae_ver),
                                      'checkpoints', 'last.ckpt')

    exp1 = SMPLexperiment(smpl_vae_model, config_smpl_vae['exp_params'])
    smpl_vae_model = exp1.load_from_checkpoint(smpl_trained_fname, vae_model=smpl_vae_model,
                                               params=config_smpl_vae['exp_params'])
    smpl_vae_model.eval()

    # load Convo_VAE model
    conv_model = vae_models[config_conv_vae['model_params']['name']](**config_conv_vae['model_params'])
    conv_trained_fname = os.path.join(config_conv_vae['logging_params']['save_dir'],
                                      config_conv_vae['model_params']['name'], 'version_{}'.format(conv_vae_ver),
                                      'checkpoints', 'last.ckpt')
    exp2 = VAEXperiment(conv_model, config_conv_vae['exp_params'])
    conv_model = exp2.load_from_checkpoint(conv_trained_fname, vae_model=conv_model,
                                           params=config_conv_vae['exp_params'])
    conv_model.eval()

    # test model with test dataset
    batch_size = 1
    tw = config_conv_vae['model_params']['tw']
    file_path = '/data/hinguyen/smpl_dataset/DIP_IMU_and_Others/'
    test_dataset = IMUDataset(file_path, tw=tw, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    chosen_idx = 856
    test_len = len(list(iter(test_loader)))
    print('test_len: {}'.format(test_len))
    e = list(iter(test_loader))[chosen_idx]

    # e = next(iter(test_examples))
    (imu, gt) = e  # imu: (b, 1, 102, tw), gt: [b, 1, 72, tw]
    y_hat = smpl_vae_model(imu, labels=gt)[0]
    print('y_hat: {}'.format(y_hat.size()))
    # # if ground_truth:
    #     smpl_forward(imu_data, gt, smpl_vae_model, batch_size, time_window, animation)

    # test ConvoVAE model with reconstructed data
    A = exp2.A
    m = exp2.m
    n = exp2.n

    imu_data_new = torch.reshape(imu, [batch_size, n])
    noise = exp2.noise_std * torch.randn(batch_size, m)
    y_batch = torch.matmul(imu_data_new, A) + noise
    y_batch = torch.reshape(y_batch, [batch_size, 1, config_conv_vae['exp_params']['h_in'], tw])
    recons = conv_model(y_batch, A=A)[0]
    print('recons: {}'.format(recons.size()))
    if not ground_truth:
        smpl_forward(recons, gt, smpl_vae_model, body_model, batch_size, tw, animation)
        smpl_forward(imu, gt, smpl_vae_model, body_model, batch_size, tw, animation)
        # Visualize the ground truth
        pose = np.transpose(gt.data.numpy(), (0, 3, 2, 1))  # [batch_size, tw, 72, 1]
        pose = torch.from_numpy(pose)
        pose = torch.reshape(pose, [batch_size * tw, config_smpl_vae['model_params']['h_out']])
        batsz = pose.shape[0]
        faces = body_model.faces
        vts, jts = model_forward(body_model, pose, batsz)
        # Visualize
        body_from_vertices(vts, faces, animation)


if __name__ == '__main__':
    eval_model(conv_vae_ver=116, spml_vae_ver=4, ground_truth=False, animation=False)