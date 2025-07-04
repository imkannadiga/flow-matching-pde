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

def evaluate_model(model, cfg: DictConfig):
    model.eval()
    device = cfg.evaluation.device
    model.to(device)  # <-- move model to correct device

    n_steps = cfg.evaluation.n_steps

    test_dataset, loader_te = load_testing_data(cfg)
    sample_idx = torch.randint(0, len(test_dataset), (1,)).item()
    sample = test_dataset[sample_idx].to(device)

    gt_sequence = sample[:n_steps].to(device)

    x0 = gt_sequence[0].unsqueeze(0).to(device)

    pred_sequence, mse_per_step = get_pred_seq(x0, gt_sequence, n_steps, model, device)

    pred_sequence = torch.stack(pred_sequence)  # [T, C, H, W]

    gt_sequence = gt_sequence.cpu()
    pred_sequence = pred_sequence.cpu()

    plot_sequence(gt_sequence, pred_sequence, cfg, step_indices=[0, 10, 15, 30, 45, 49])

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_steps), mse_per_step, marker='o')
    plt.title('MSE per Step')
    plt.xlabel('Step')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid()
    eval_path = Path(cfg.evaluation.eval_path)
    eval_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(eval_path / 'mse_per_step.png')
    plt.close()

    compare_spectra(gt_sequence, pred_sequence, eval_path / 'spectrum.png')

    real = []
    gen = []

    with torch.no_grad():
        for batch, target in loader_te:
            batch = batch.to(device)
            target = target.to(device)
            output = model(batch)
            gen.append(output)
            real.append(target)

    real = torch.cat(real, dim=0).squeeze(1).cpu()
    gen = torch.cat(gen, dim=0).squeeze(1).cpu()

    print("Computing evaluation metrics...")
    dens_mse = density_mse(real, gen)
    print(f"Density MSE: {dens_mse:.4e}")

    distribution_kde(real, gen, save_path=eval_path / "distribution_kde.png")
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

    eval_path = Path(cfg.evaluation.eval_path)
    eval_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(eval_path / 'evaluation_sequence.png')
    return

def get_pred_seq(x0, x_act, n_steps, model, device):
    pred_sequence = [x0.squeeze(0).detach()]
    mse_per_step = []

    x_curr = x0.clone()
    for step in range(1, n_steps):
        with torch.no_grad():
            x_next = model(x_curr.to(device))  # ensure input is on correct device

        pred_sequence.append(x_next.squeeze(0).detach())

        gt = x_act[step].to(device)
        mse = F.mse_loss(x_next.squeeze(0), gt)
        mse_per_step.append(mse.item())

        x_curr = x_next.detach()

    return pred_sequence, mse_per_step
