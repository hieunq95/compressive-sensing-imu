## Generative model-based compressive sensing for IMU data

#### 1. Install packages
`pip install -r requirements.txt`

#### 2. Download datasets
##### 2.1. IMU dataset
Download DIP-IMU dataset (DIP_IMU.zip) from [DIP project website](https://dip.is.tuebingen.mpg.de/index.html).

Unzip the file, modify the parameters `file_path` in `eval_vae.py',  and `data_path` in `configs/*.yaml` files.

##### 2.2. SMPL dataset
Download the SMPL dataset from [SMPL project website](https://smpl.is.tue.mpg.de/download.php).

Select version 1.0.0 for Python 2.7 (female/male. 10 shape PCs)

#### 3. Reproduce results
##### 3.1. Training
Run `train_mlp_vae.py` to train CS-VAE model.

Run `train_smpl_vae.py` to train a VAE model for mapping IMU signals (204 features) to SMPL pose (72 features).

Run `train_dip.py` to train the DIP baseline.

All scenarios can be obtained by changing parameters `h_in`, `h_out`, and `eta` in yaml files.

After training, the training results should be placed in directories like `logs/VanillaVAE`, `logs/DIPVAE`, 
`logs/SMPLVAE`, and `DecodingTime`.

The running versions are automatically named by pytorch-lighting.

##### 3.2. Plot results
We use `eval_vae.py` to evaluate and plot results. 

Make sure that `file_path` and `bm_fname` parameters in this file are correct.

Change parameter `EVALUATION` to True and run the file to obtain all results. 

If you only want to plot the results, set the `EVALUATION` parameter to False and rerun the fil.
