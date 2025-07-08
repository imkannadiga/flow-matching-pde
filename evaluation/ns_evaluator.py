import os
from omegaconf import DictConfig
import torch
from matplotlib import pyplot as plt
from evaluation.base import BaseEvaluator
from evaluation.eval_utils import (
    get_pred_seq,
    plot_sequence,
    density_mse,
    distribution_kde,
    compare_spectra,
)
from data.navier_stokes import NSDataModule

class NSEvaluator(BaseEvaluator):
    def _load_dataset(self):
        dataset = NSDataModule(**self.cfg.eval.data)
        _, loader_te = dataset.get_testing_data()
        return dataset, loader_te

    def run(self):
        print("[Evaluator] Running Navier-Stokes evaluation...")
        self.visualize_sample()
        self.full_dataset_metrics()
        print("[Evaluator] Evaluation complete.")

    def visualize_sample(self):
        cfg = self.cfg
        device = self.device

        # Get a random sample
        test_dataset, _ = self.dataset.get_testing_data()
        idx = torch.randint(0, len(test_dataset), (1,)).item()
        sample = test_dataset[idx].to(device)

        gt_sequence = sample[:cfg.eval.n_steps]
        x0 = gt_sequence[0].unsqueeze(0)

        pred_sequence, mse_per_step = get_pred_seq(x0, gt_sequence, cfg.eval.n_steps, self.model, device)

        plot_sequence(gt_sequence.cpu(), torch.stack(pred_sequence).cpu(), cfg)

        plt.figure()
        plt.plot(range(1, cfg.eval.n_steps), mse_per_step, marker="o")
        plt.title("MSE per Step")
        plt.xlabel("Step")
        plt.ylabel("MSE")
        os.makedirs(cfg.eval.eval_path, exist_ok=True)
        plt.savefig(f"{cfg.eval.eval_path}/mse_per_step.png")
        plt.close()

        compare_spectra(gt_sequence.cpu(), torch.stack(pred_sequence).cpu(), f"{cfg.eval.eval_path}/spectrum.png")

    def full_dataset_metrics(self):
        real, gen = [], []
        with torch.no_grad():
            for batch in self.loader_te:
                batch["x"], batch["y"] = batch["x"].to(self.device), batch["y"].to(self.device)
                output = self.model(batch["x"])
                gen.append(output)
                real.append(batch["y"])

        real = torch.cat(real, dim=0).squeeze(1).cpu()
        gen = torch.cat(gen, dim=0).squeeze(1).cpu()

        print(f"[Evaluator] Density MSE: {density_mse(real, gen):.4e}")
        distribution_kde(real, gen, f"{self.cfg.eval.eval_path}/distribution_kde.png")
