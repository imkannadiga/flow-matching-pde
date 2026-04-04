import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Union
import numpy as np
from scipy.stats import gaussian_kde
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_pred_seq_with_comparision(test_loader, n_steps, model, device, save_path):
    logger.info("Starting prediction and comparison...")
    model.eval()
    pred_seq = []
    mse_per_timestep = torch.zeros(n_steps - 1, device=device)
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}...")
            batch = batch.to(device)
            x = batch.clone()
            
            batch_size = x.shape[0]
            spatial_shape = x.shape[2:]

            preds = []

            x_t = x[:, 0]
            preds.append(x_t.cpu())

            for t in range(1, n_steps):
                x_t = sample(x0=x_t, model=model, device=device)
                preds.append(x_t.cpu())

                mse = F.mse_loss(x_t, x[:, t], reduction='mean')
                mse_per_timestep[t - 1] += mse

                logger.debug(f"Batch {batch_idx + 1}, timestep {t}: MSE = {mse.item():.6f}")

            num_batches += 1
            pred_seq.append(torch.stack(preds, dim=1))

    pred_seq = torch.cat(pred_seq, dim=0)
    mse_per_timestep /= num_batches

    logger.info("Finished all batches.")
    logger.info("Plotting MSE per timestep...")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_steps), mse_per_timestep.cpu().numpy(), marker='o')
    plt.title('MSE vs Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / 'evaluation_mse_per_step.png')
    plt.close()
    
    result_path = save_path / 'results.pt'
    torch.save({
        'pred_seq': pred_seq, 
        'mse_per_timestep': mse_per_timestep
    }, result_path)

    logger.info(f"Saved predictions and MSE to {result_path}")

    return pred_seq, mse_per_timestep


def sample(x0, model, device='cpu', return_path=False, rtol=1e-5, atol=1e-5):
    # x0: samples at time t
    # model: flow model trained on PDE
    # n_eval: how many timesteps in [0, 1] to evaluate. Should be >= 2. 
    # returns sample at time t+1
    from torchdiffeq import odeint

    t = torch.linspace(0, 1, model.t_scaling, device=device)
    
    method = 'dopri5'
    out = odeint(model, x0, t, method=method, rtol=rtol, atol=atol)

    if return_path:
        return out
    else:
        return out[-1]


def get_pred_seq(x0, gt_sequence, n_steps, model, device, include_time=False):
    """
    Generate a predicted sequence using the model starting from x0.
    
    Args:
        x0: Initial condition tensor [1, C, H, W]
        gt_sequence: Ground truth sequence tensor [n_steps, C, H, W]
        n_steps: Number of steps to predict
        model: Trained model
        device: Device to run on
        include_time: Whether the model requires time input (for flow matching models)
    
    Returns:
        pred_sequence: List of predicted tensors
        mse_per_step: MSE for each step compared to ground truth
    """
    model.eval()
    pred_sequence = []
    mse_per_step = []
    
    # Handle x0 shape - ensure it's [1, C, H, W]
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    
    current_x = x0.to(device)
    pred_sequence.append(current_x.cpu().squeeze(0))
    
    with torch.no_grad():
        for step in range(1, n_steps):
            if include_time:
                # For flow matching models, we need to provide time
                # Use step index normalized to [0, 1]
                t = torch.tensor([step / n_steps], device=device)
                # All models now use consistent signature: forward(t, u)
                # Try (t, u) first (consistent across all models)
                try:
                    pred = model(t, current_x)
                except TypeError:
                    # Fallback to (u, t) if needed for backward compatibility
                    pred = model(current_x, t)
            else:
                # For non-flow-matching models, just use current_x
                if current_x.dim() == 4:
                    # Model expects dict input with "x" key or direct tensor
                    try:
                        pred = model(current_x)
                    except:
                        # Try as dict
                        pred = model(**{"x": current_x})
                else:
                    pred = model(current_x)
            
            # Store prediction
            pred_sequence.append(pred.cpu().squeeze(0) if pred.dim() > 3 else pred.cpu())
            
            # Calculate MSE if ground truth is available
            if step < len(gt_sequence):
                gt_step = gt_sequence[step].unsqueeze(0).to(device) if gt_sequence[step].dim() == 3 else gt_sequence[step].to(device)
                mse = F.mse_loss(pred, gt_step).item()
                mse_per_step.append(mse)
                
                # For autoregressive prediction, use prediction as next input
                current_x = pred
            else:
                # No ground truth, just use prediction
                current_x = pred
    
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