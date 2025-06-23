import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from data import load_raw_data
from omegaconf import DictConfig

def evaluate_model(model, cfg:DictConfig):
    # === CONFIGURATION ===
    n_steps = cfg.evaluation.n_steps
    device = cfg.evaluation.device
    
    # === LOAD DATA ===
    dataset = load_raw_data(cfg)  # Replace path as needed
    sample_idx = torch.randint(0, len(dataset), (1,)).item()
    sample = dataset[sample_idx].to(device)  # Shape: [T, C, H, W]
    sample = sample.unsqueeze(1)
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
        x_next = x_next.unsqueeze(0)  # Add batch dimension

        print("2 :: ",x_next.shape)
        
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
