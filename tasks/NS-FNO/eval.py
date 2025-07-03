import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

os.environ['WANDB_MODE'] = "disabled"

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from data import load_testing_data
from omegaconf import DictConfig
from util.eval import distribution_kde, compare_spectra, density_mse, spectra_mse

def evaluate_model(model, cfg:DictConfig):
    # Put model to evaluation mode
    model.eval()
    # === CONFIGURATION ===
    n_steps = cfg.evaluation.n_steps
    device = cfg.evaluation.device
        
    # === LOAD DATA ===
    test_dataset, loader_te = load_testing_data(cfg)  # Replace path as needed
    sample_idx = torch.randint(0, len(test_dataset), (1,)).item()
    sample = test_dataset[sample_idx].to(device)  # Shape: [T, C, H, W]
    
    gt_sequence = sample[:n_steps]  # Ground truth (25 frames)
    
    # === INITIAL CONDITION ===
    x0 = gt_sequence[0].unsqueeze(0)  # Shape: [1, C, H, W]
    
    pred_sequence, mse_per_step = get_pred_seq(x0, gt_sequence, n_steps, model)
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
    
    # Spectrum will be computed on the generated sequence of images
    compare_spectra(gt_sequence, pred_sequence, eval_path/'spectrum.png')
    
    # For density, we create test samples as done before and get the model prediction
    
    real = []
    gen = []
    
    with torch.no_grad():
        for batch, target in loader_te:
            gen.append(model(batch).detach().cpu())
            real.append(target.detach().cpu())

    real = torch.cat(real, dim=0).squeeze(1)  # Shape: (N, H, W)
    gen = torch.cat(gen, dim=0).squeeze(1)
    
    # === Run evaluation ===
    print("Computing evaluation metrics...")
    dens_mse = density_mse(real, gen)
    print(f"Density MSE: {dens_mse:.4e}")
    # Optional: save evaluation plots
    distribution_kde(real, gen, save_path = eval_path / "distribution_kde.png")
    print("Evaluation complete")
    
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

def get_pred_seq(x0, x_act, n_steps, model):
    pred_sequence = [x0.squeeze(0)]
    mse_per_step = []

    x_curr = x0.clone()
    for step in range(1, n_steps):
        # Sample next frame
        with torch.no_grad():
            x_next = model(x_curr)

        # Save prediction
        pred_sequence.append(x_next.squeeze(0).detach())

        # Compute MSE
        gt = x_act[step]
        mse = F.mse_loss(x_next.squeeze(0), gt)
        mse_per_step.append(mse.item())

        # Prepare for next step
        x_curr = x_next.detach()
        
    return pred_sequence, mse_per_step
