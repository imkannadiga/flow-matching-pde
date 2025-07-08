import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from data.navier_stokes import NSDataModule
from evaluation.eval_utils import load_model_from_manifest,get_pred_seq, plot_sequence, density_mse, distribution_kde, compare_spectra
from matplotlib import pyplot as plt
import os

def evaluate(cfg: DictConfig):
    
    device = cfg.eval.device
    
    # TODO: Load model from latest training run
    model = load_model_from_manifest(cfg.train.save_path, model_raw=instantiate(cfg.model))
    model.eval()
    model.to(device)

    # Select a sample for visualization
    dataset = NSDataModule(cfg=cfg)
    test_dataset, loader_te = dataset.get_testing_data()
    sample_idx = torch.randint(0, len(test_dataset), (1,)).item()
    sample = test_dataset[sample_idx].to(device)

    gt_sequence = sample[:cfg.eval.n_steps]
    x0 = gt_sequence[0].unsqueeze(0)

    pred_sequence, mse_per_step = get_pred_seq(x0, gt_sequence, cfg.eval.n_steps, model, device)

    plot_sequence(gt_sequence.cpu(), torch.stack(pred_sequence).cpu(), cfg)

    # MSE Curve
    plt.figure()
    plt.plot(range(1, cfg.eval.n_steps), mse_per_step, marker="o")
    plt.title("MSE per Step")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    os.makedirs(cfg.eval.eval_path, exist_ok=True)
    plt.savefig(f"{cfg.eval.eval_path}/mse_per_step.png")
    plt.close()

    # Spectral comparison
    compare_spectra(gt_sequence.cpu(), torch.stack(pred_sequence).cpu(), f"{cfg.eval.eval_path}/spectrum.png")

    # Full dataset evaluation
    real, gen = [], []
    with torch.no_grad():
        for batch in loader_te:
            batch["x"], batch["y"] = batch["x"].to(device), batch["y"].to(device)
            output = model(batch)
            gen.append(output)
            real.append(batch["y"])

    real = torch.cat(real, dim=0).squeeze(1).cpu()
    gen = torch.cat(gen, dim=0).squeeze(1).cpu()

    print(f"Density MSE: {density_mse(real, gen):.4e}")
    distribution_kde(real, gen, f"{cfg.eval.eval_path}/distribution_kde.png")
    print("Evaluation complete")

