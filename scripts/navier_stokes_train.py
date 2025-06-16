# python script to train and test a FFM model on the navier stokes dataset
import sys
sys.path.append('../')

from util.config import load_config

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
from util.gaussian_process import GPPrior
from util.util import download_dataset, make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples
import time

from util.util import load_navier_stokes
from torch.utils.data import TensorDataset, DataLoader

from models.fno import FNO
from models.FFM import FFMModel

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from util.eval import density_mse, spectra_mse, compare_spectra, distribution_kde

config = load_config('../config.yaml')

def load_and_prepare(data_identifier):
    print(f'Loading data from {data_identifier}')
    data = torch.from_numpy(np.load(data_identifier))
    print(f'Loaded data with shape {data.shape}')
    data = data.permute(3, 1, 2, 0)
    data = data.permute(0, -1, 1, 2).contiguous().reshape(-1, 64, 64).unsqueeze(1) 
    idx = torch.randperm(data.shape[0])
    data = data[idx]
    print(f'Processed data to shape {data.shape}')
    data_tr, data_te = data[:config['ntr']], data[config['ntr']:]
    loader_tr = DataLoader(data_tr, batch_size=config['batch_size'], shuffle=True)
    loader_te = DataLoader(data_te, batch_size=config['batch_size'], shuffle=True)

    return loader_tr, loader_te

def create_model():
    print('Initializing model....')
    model = FNO(config['modes'], config['visch'], config['hch'], config['pch'], x_dim=config['xdim'], t_scaling=config['t_scaling'])
    model.to(config['device'])
    print(model)
    model_wrapper = FFMModel(model, 
                         kernel_length=config['lengthscale'], 
                         kernel_variance=config['var'], 
                         sigma_min=config['sigma_min'], 
                         device=config['device'])
    return model, model_wrapper

data_file = download_dataset(config['download_url'], config['download_source'])

loader_tr, loader_te = load_and_prepare(data_file)

model, model_wrapper = create_model()

optimizer = Adam(model.parameters(), config['lr'])
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

print('Starting training....')

model_wrapper.train(loader_tr,
                    optimizer,
                    epochs=config['nepoch'],
                    scheduler=scheduler,
                    test_loader=loader_te,
                    eval_int=config['evalint'],
                    save_int=config['saveint'],
                    generate=config['generate'],
                    save_path=Path(config['spath'])
                    )

print("Training completed, performing evaluation....")

model_wrapper.eval()
test_iter = iter(loader_te)
test_batch = next(test_iter)  # shape: (batch_size, 1, 64, 64)
test_batch = test_batch.to(config['device'])

with torch.no_grad():
    # Get model prediction
    pred = model(test_batch)  # Assumes model returns shape: (batch_size, 1, 64, 64)

# === Convert shapes to (N, 64, 64) ===
real = test_batch.squeeze(1).cpu()  # Remove channel dim
gen = pred.squeeze(1).cpu()

# === Run evaluation ===
print("Computing evaluation metrics...")

dens_mse = density_mse(real, gen)
spec_mse = spectra_mse(real, gen)

print(f"Density MSE: {dens_mse:.4e}")
print(f"Spectrum MSE: {spec_mse:.4e}")

# Optional: save evaluation plots
distribution_kde(real, gen, save_path="../evaluation/distribution_kde.png")
compare_spectra(real, gen, save_path="../evaluation/spectrum_comparison.png")

print("Evaluation complete. Plots saved.")