import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

os.environ['WANDB_MODE'] = "disabled"

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from data import load_raw_data
from omegaconf import DictConfig
from util.eval import distribution_kde, compare_spectra, density_mse, spectra_mse

def evaluate_model(model, cfg:DictConfig, loader_te):
    # === CONFIGURATION ===
    n_steps = cfg.evaluation.n_steps
    device = cfg.evaluation.device
    
    # === LOAD DATA ===
    dataset = load_raw_data(cfg)  # Replace path as needed
    sample_idx = torch.randint(0, len(dataset), (1,)).item()
    sample = dataset[sample_idx].to(device)  # Shape: [T, C, H, W]
    print("0 :: ",sample.shape)
    gt_sequence = sample[:n_steps]  # Ground truth (25 frames)
    
    print("1 :: ",gt_sequence.shape)
    
    # === INITIAL CONDITION ===
    x0 = gt_sequence[0].unsqueeze(0)  # Shape: [1, C, H, W]
    
    print("2 :: ",x0.shape)
    
    dims = list(x0.shape[2:])         # e.g. [64, 64]
    n_channels = x0.shape[1]
    
    # === SAMPLING LOOP ===
    pred_sequence = [x0.squeeze(0)]
    mse_per_step = []

    x_curr = x0.clone()
    for step in range(1, n_steps):
        # Sample next frame
        print("3 :: ",x_curr.shape)
        x_next = model.sample(dims=dims, x0=x_curr, n_channels=n_channels, n_eval=2)

        print("4 :: ",x_next.shape)
        
        # Save prediction
        pred_sequence.append(x_next.squeeze(0).detach())

        # Compute MSE
        gt = gt_sequence[step]
        mse = F.mse_loss(x_next.squeeze(0), gt)
        mse_per_step.append(mse.item())

        # Prepare for next step
        x_curr = x_next.detach()

    # === STACK RESULTS ===
    pred_sequence = torch.stack(pred_sequence)  # [T, C, H, W]
    gt_sequence = gt_sequence.cpu()
    pred_sequence = pred_sequence.cpu()

    # === PLOTTING ===
    plot_sequence(gt_sequence, pred_sequence, cfg, step_indices=[0, 5, 10, 15, 20, 24])

    # plot mse sequence
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_steps), mse_per_step, marker='o')
    plt.title('MSE per Step')
    plt.xlabel('Step')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid()
    eval_path = Path() / cfg.evaluation.eval_path
    if not eval_path.exists():
        eval_path.mkdir(parents=True)
    plt.savefig(eval_path / 'mse_per_step.png')
    plt.close()
    
    # Density and Spectrum Evaluation
    
    test_iter = iter(loader_te)
    test_batch = next(test_iter)  # shape: (batch_size, 1, 64, 64)
    test_batch = test_batch.to(cfg.evaluation.device)

    with torch.no_grad():
        # Get model prediction
        pred = model.sample(dims=test_batch.shape[2:], 
                            n_channels=test_batch[1],
                            x0=test_batch,
                            n_eval=1000)  # Assumes model returns shape: (batch_size, 1, 64, 64)

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
    distribution_kde(real, gen, save_path = eval_path / "distribution_kde.png")
    compare_spectra(real, gen, save_path = eval_path / "spectrum_comparison.png")

    print("Evaluation complete")
    
    distribution_kde(real, gen, save_path = eval_path / "distribution_kde.png")
    compare_spectra(real, gen, save_path = eval_path / "spectrum_comparison.png")
    
    return None

def plot_sequence(gt, pred, cfg, step_indices=None):
    if step_indices is None:
        step_indices = list(range(len(gt)))
    fig, axes = plt.subplots(len(step_indices), 2, figsize=(6, 3 * len(step_indices)))
    for i, idx in enumerate(step_indices):
        gt_img = gt[idx][0].numpy()
        pred_img = pred[idx][0].numpy()
        axes[i, 0].imshow(gt_img, cmap='viridis')
        axes[i, 0].set_title(f"Ground Truth Step {idx}")
        axes[i, 1].imshow(pred_img, cmap='viridis')
        axes[i, 1].set_title(f"Prediction Step {idx}")
        
    # save the plot inside the evaluation path
    eval_path = Path(cfg.evaluation.eval_path)
    if not eval_path.exists():
        eval_path.mkdir(parents=True)
    plt.savefig(eval_path / 'evaluation_sequence.png')
    return
