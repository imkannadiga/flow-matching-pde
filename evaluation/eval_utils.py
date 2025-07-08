import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Union
import numpy as np
from scipy.stats import gaussian_kde

def load_model_from_manifest(save_dir: Union[str, Path], model_raw: torch.nn.Module):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # Load the manifest file
    manifest = torch.load(save_dir / "manifest.pt", map_location='cpu')

    # Path to the saved model state dict
    model_path = save_dir / manifest['model']

    # Instantiate model and load weights
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model_raw.load_state_dict(state_dict)

    return model_raw

def get_pred_seq(x0, x_act, n_steps, model, device):
    pred_sequence = [x0.squeeze(0).detach()]
    mse_per_step = []

    x_curr = x0.clone().to(device)
    for step in range(1, n_steps):
        with torch.no_grad():
            x_next = model(x_curr)
        pred_sequence.append(x_next.squeeze(0).detach())
        gt = x_act[step].to(device)
        mse = F.mse_loss(x_next.squeeze(0), gt)
        mse_per_step.append(mse.item())
        x_curr = x_next.detach()

    return pred_sequence, mse_per_step


def plot_sequence(gt, pred, cfg, step_indices=None):
    if step_indices is None:
        step_indices = [0, 10, 15, 30, 45, 49]

    fig, axes = plt.subplots(len(step_indices), 2, figsize=(6, 3 * len(step_indices)))
    for i, idx in enumerate(step_indices):
        gt_img = gt[idx][0].numpy()
        pred_img = pred[idx][0].numpy()
        axes[i, 0].imshow(gt_img, cmap='viridis')
        axes[i, 0].set_title(f"Ground Truth Step {idx}")
        axes[i, 1].imshow(pred_img, cmap='viridis')
        axes[i, 1].set_title(f"Prediction Step {idx}")
    os.makedirs(cfg.eval.eval_path, exist_ok=True)
    plt.savefig(f"{cfg.eval.eval_path}/evaluation_sequence.png")
    plt.close()

def distribution_kde(real, gen, n=1000, bw=0.5, save_path=None, gridsize=200, cutoff=3):
    # Compares the distribution of pointwise values between real and generated
    # u is data, n is how large of a subset to look at (useful for very large # of samples)
    
    real = real.flatten()
    idx = torch.randperm(real.numel())
    real = real[idx[:n]]
    
    gen = gen.flatten()
    idx = torch.randperm(gen.numel())
    gen = gen[idx[:n]]

    kernel_real = gaussian_kde(real, bw_method=bw)
    kernel_gen = gaussian_kde(gen, bw_method=bw)

    min_val = torch.min(torch.min(real), torch.min(gen))
    max_val = torch.max(torch.max(real), torch.max(gen))

    min_cutoff = min_val - cutoff * bw * np.abs(min_val)
    max_cutoff = max_val + cutoff * bw * np.abs(max_val)

    grid = torch.linspace(min_cutoff, max_cutoff, gridsize)

    y_real = kernel_real(grid)
    y_gen = kernel_gen(grid)

    mse = np.mean((y_real - y_gen)**2)

    fig, ax = plt.subplots()
    ax.plot(grid, y_real, label='Ground Truth')
    ax.plot(grid, y_gen, label='Generated', linestyle='--')

    ax.legend()

    ax.set_title(f'MSE: {mse:.4e}')
    
    plt.legend()
    if save_path:
        plt.savefig(save_path)

def spectrum(u, s):
    # https://github.com/neuraloperator/markov_neural_operator/blob/main/visualize_navier_stokes2d.ipynb
    # s is the resolution of u, i.e. u is shape (batch_size, s, s)

    T = u.shape[0]
    u = u.reshape(T, s, s)
    u = torch.fft.fft2(u)

    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers
    
    # wavenumers = [0, 1, 2, n/2-1, -n/2, -n/2 + 1, ..., -3, -2, -1]
    
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y
                                       )
    sum_k = sum_k.numpy()
    
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
    
    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        spectrum[:, j - 1] = np.sqrt( (u[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2)
        
    spectrum = spectrum.mean(axis=0)
    spectrum = spectrum[:s//2]

    return spectrum


def compare_spectra(real, gen, save_path=None):
    s = real.shape[-1]
    spec_true = spectrum(real, s)
    spec_gen = spectrum(gen, s)

    mse = np.mean( (spec_true-spec_gen)**2 )

    fig, ax = plt.subplots()

    ax.semilogy(spec_true, label='Ground Truth')
    ax.semilogy(spec_gen, label='Generated')
    ax.legend()

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Energy')

    ax.set_title(f'MSE: {mse:4e}')

    if save_path:
        plt.savefig(save_path)    


def spectra_mse(real, gen):
    s = real.shape[-1]
    spec_true = spectrum(real, s)
    spec_gen = spectrum(gen, s)

    mse = np.mean( (spec_true-spec_gen)**2 )
    return mse

def density_mse(real, gen, bw=0.5, gridsize=200, cutoff=3):
    # Compares the distribution of pointwise values between real and generated
    # u is data, n is how large of a subset to look at (useful for very large # of samples)

    real = real.flatten()
    gen = gen.flatten()

    kernel_real = gaussian_kde(real, bw_method=bw)
    kernel_gen = gaussian_kde(gen, bw_method=bw)

    min_val = torch.min(torch.min(real), torch.min(gen))
    max_val = torch.max(torch.max(real), torch.max(gen))

    min_cutoff = min_val - cutoff * bw * np.abs(min_val)
    max_cutoff = max_val + cutoff * bw * np.abs(max_val)

    grid = torch.linspace(min_cutoff, max_cutoff, gridsize)

    y_real = kernel_real(grid)
    y_gen = kernel_gen(grid)

    mse = np.mean((y_real - y_gen)**2)

    return mse