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
from data.navier_stokes_fm import NSFMDataModule

class NSEvaluator(BaseEvaluator):
    def _load_dataset(self):
        if getattr(self.cfg.eval, "include_time", False):
            data_module = NSFMDataModule(**self.cfg.eval.data)
        else:
            data_module = NSDataModule(**self.cfg.eval.data)

        dataset, loader_te = data_module.get_testing_data()
        return dataset, loader_te

    def run(self):
        print("[Evaluator] Running Navier-Stokes evaluation...")
        self.visualize_sample()
        self.full_dataset_metrics()
        print("[Evaluator] Evaluation complete.")

    def visualize_sample(self):
        cfg = self.cfg
        device = self.device

        test_dataset = self.dataset
        idx = torch.randint(0, len(test_dataset), (1,)).item()
        sample = test_dataset[idx].to(device)

        gt_sequence = sample[:cfg.eval.n_steps]
        x0 = gt_sequence[0].unsqueeze(0)

        include_time = getattr(cfg.eval, "include_time", False)
        pred_sequence, mse_per_step = get_pred_seq(
            x0, gt_sequence, cfg.eval.n_steps, self.model, device, include_time=include_time
        )

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
        include_time = getattr(self.cfg.eval, "include_time", False)

        with torch.no_grad():
            for batch in self.loader_te:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if include_time:
                    output = self.model(x_t=batch["x"], t=batch["t"])
                else:
                    output = self.model(batch["x"])
                gen.append(output)
                real.append(batch["y"])

        real = torch.cat(real, dim=0).squeeze(1).cpu()
        gen = torch.cat(gen, dim=0).squeeze(1).cpu()

        print(f"[Evaluator] Density MSE: {density_mse(real, gen):.4e}")
        distribution_kde(real=real, gen=gen, save_path=f"{self.cfg.eval.eval_path}/distribution_kde.png")
