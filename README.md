## Generative model-based compressive sensing for IMU data

#### 1. Install packages
`pip install -r requirements.txt`

Python version: 3.8

#### 2. Download datasets
##### 2.1. IMU dataset
Download DIP-IMU dataset (DIP_IMU_and_Others.zip) from [DIP project website](https://dip.is.tuebingen.mpg.de/index.html).

Unzip the file `DIP_IMU_and_Others`, modify the parameters `file_path` in [`eval_vae.py`](eval_vae.py),  and `data_path` in [`configs`](configs)/*.yaml files.

##### 2.2. SMPL dataset
Download the SMPL dataset from [SMPL project website](https://smpl.is.tue.mpg.de/download.php).

Select version 1.0.0 for Python 2.7 (female/male. 10 shape PCs).

Unzip the file, modify the parameters `bm_fname` in [`eval_vae.py`](eval_vae.py).

#### 3. Reproduce results
##### 3.1. Pre-processing
Before training the models, we need to pre-processing the IMU dataset.

For this, let's change the parameter `processing` in the class `IMUDataset` in [`dataset.py`](dataset.py) to True at your first training.

The pre-processed datasets are then saved into npz files in directory `DIP_IMU_and_Others`.

Note that we need to pre-process the dataset only once. From the second training, `processing` parameter should be changed 
to False for faster training.

##### 3.2. Training
CPU and GPU training: modify parameter `gpus` in the [`configs`](configs)/*.yaml files. 
For CPU training, the setting is `gpus: []`.
For single GPU training, the setting is `gpus: [0]`.

Run [`train_mlp_vae.py`](train_mlp_vae.py) to train CS-VAE model.

Run [`train_smpl_vae.py`](train_smpl_vae.py) to train a VAE model for mapping IMU signals (204 features) to SMPL pose (72 features).

Run [`train_dip.py`](train_dip.py) to train the DIP baseline.

All scenarios can be obtained by changing parameters `h_in` and `eta` in yaml files.

After training, the training results should be placed in directories like `logs/VanillaVAE`, `logs/DIPVAE`, 
`logs/SMPLVAE`, and `logs/DecodingTime`.

The running versions are automatically named by pytorch-lightning.

##### 3.3. Plot results
We use [`eval_vae.py`](eval_vae.py) to evaluate and plot results. 

Make sure that `file_path` and `bm_fname` parameters in this file are correct.

Change parameter `EVALUATION` to True and run the file to obtain all results. 

If you only want to plot the results, set the `EVALUATION` parameter to False and rerun the file.
