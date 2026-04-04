import os
from omegaconf import DictConfig
import torch
from matplotlib import pyplot as plt
from hydra.utils import instantiate
from evaluation.base import BaseEvaluator
from evaluation.eval_utils import (
    get_pred_seq,
    plot_sequence,
    density_mse,
    distribution_kde,
    compare_spectra,
    load_model_from_manifest,
)
from data.navier_stokes import NSDataModule
from data.navier_stokes_fm import NSFMDataModule
from pathlib import Path

class NSEvaluator(BaseEvaluator):
    def __init__(self, cfg: DictConfig):
        """
        Initialize NSEvaluator with config.
        
        Args:
            cfg: Hydra config containing model, data, eval, and other settings
        """
        self.cfg = cfg
        self.device = torch.device(cfg.eval.device if hasattr(cfg.eval, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = instantiate(cfg.model)
        state_dict_path = cfg.eval.state_dict_path if hasattr(cfg.eval, 'state_dict_path') else cfg.train.save_path
        if state_dict_path:
            save_dir = Path(state_dict_path)
            if (save_dir / "manifest.pt").exists():
                self.model = load_model_from_manifest(save_dir, self.model)
            elif (save_dir / "model_state_dict.pt").exists():
                state_dict = torch.load(save_dir / "model_state_dict.pt", map_location='cpu', weights_only=False)
                self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load dataset
        self.dataset, self.loader_te = self._load_dataset()
    
    def _load_dataset(self):
        if getattr(self.cfg.eval, "include_time", False):
            data_module = NSFMDataModule(**self.cfg.eval.data)
        else:
            data_module = NSDataModule(**self.cfg.eval.data)

        # Get testing data - check if n_samples is specified in config
        n_samples = getattr(self.cfg.eval, 'n_samples', None)
        dataset, loader_te = data_module.get_testing_data(n_samples=n_samples, data_loader=True)
        return dataset, loader_te

    def run(self):
        print("[Evaluator] Running Navier-Stokes evaluation...")
        self.visualize_sample()
        self.full_dataset_metrics()
        print("[Evaluator] Evaluation complete.")

    def visualize_sample(self):
        cfg = self.cfg
        device = self.device

        # Get a sample from the dataset
        test_dataset = self.dataset
        idx = torch.randint(0, len(test_dataset), (1,)).item()
        sample = test_dataset[idx]
        
        # Handle both dict and tensor samples
        if isinstance(sample, dict):
            # If sample is a dict, get the sequence from it
            # For flow matching datasets, we may need to reconstruct sequence
            # For now, use x as starting point
            if "x" in sample:
                x0 = sample["x"].unsqueeze(0).to(device) if sample["x"].dim() == 3 else sample["x"].to(device)
            else:
                raise ValueError("Sample dict must contain 'x' key")
            
            # For visualization, we'll use the loader to get actual sequences
            # Use first batch for visualization
            batch = next(iter(self.loader_te))
            if isinstance(batch, dict):
                if "y" in batch:
                    gt_sequence = batch["y"][0].cpu()  # Get first item in batch
                else:
                    # Create dummy sequence from x
                    gt_sequence = torch.stack([x0.squeeze(0).cpu()] * cfg.eval.n_steps)
            else:
                gt_sequence = batch[0].cpu()
        else:
            # If sample is a tensor, use it directly
            sample = sample.to(device)
            if sample.dim() == 4:  # [T, C, H, W]
                gt_sequence = sample[:cfg.eval.n_steps].cpu()
                x0 = gt_sequence[0].unsqueeze(0).to(device)
            else:
                raise ValueError(f"Unexpected sample shape: {sample.shape}")

        include_time = getattr(cfg.eval, "include_time", False)
        pred_sequence, mse_per_step = get_pred_seq(
            x0, gt_sequence.to(device), cfg.eval.n_steps, self.model, device, include_time=include_time
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
                u = batch["x"].to(self.device)
                if include_time:
                    tt = batch["t"].to(self.device)
                    if tt.dim() == 0:
                        tt = tt.unsqueeze(0).expand(u.shape[0])
                    elif tt.dim() == 1 and tt.shape[0] == 1 and u.shape[0] > 1:
                        tt = tt.expand(u.shape[0])
                    output = self.model(t=tt, u=u)
                else:
                    bsz = u.shape[0]
                    t0 = torch.zeros(bsz, device=self.device, dtype=u.dtype)
                    output = self.model(t=t0, u=u)
                gen.append(output)
                real.append(batch["y"].to(self.device))

        real = torch.cat(real, dim=0).squeeze(1).cpu()
        gen = torch.cat(gen, dim=0).squeeze(1).cpu()

        print(f"[Evaluator] Density MSE: {density_mse(real, gen):.4e}")
        distribution_kde(real=real, gen=gen, save_path=f"{self.cfg.eval.eval_path}/distribution_kde.png")
