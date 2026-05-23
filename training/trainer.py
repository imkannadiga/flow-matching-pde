import sys
import warnings
from timeit import default_timer
from pathlib import Path
from typing import Any, Union, Optional
import contextlib

import torch
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    pass

from .training_state import load_training_state, save_training_state


class LpLoss(nn.Module):
    """Relative Lp loss over spatial dimensions — drop-in replacement for neuralop.losses.LpLoss."""
    def __init__(self, d=2, p=2, reduction=True, size_average=True):
        super().__init__()
        self.d = d
        self.p = p
        self.reduction = "mean" if size_average else "sum"

    def forward(self, x, y):
        dims = list(range(-self.d, 0))
        diff = torch.norm(x - y, p=self.p, dim=dims)
        norm = torch.norm(y, p=self.p, dim=dims).clamp(min=1e-8)
        return (diff / norm).mean()


def _wandb_numeric(x: Any) -> float:
    """Convert torch / numpy scalars to a Python float for W&B."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if hasattr(x, "item") and callable(x.item) and not isinstance(x, (float, int, bool)):
        try: return float(x.item())
        except Exception: pass
    return float(x)


def _wandb_histogram_param(tensor: torch.Tensor):
    """Build a W&B histogram from a parameter tensor."""
    flat = tensor.detach().cpu().float().reshape(-1)
    if flat.numel() == 0 or float(flat.max()) == float(flat.min()):
        return None
    return wandb.Histogram(flat.numpy())


class Trainer:
    """A general Trainer class to train models on given datasets."""
    
    def __init__(
        self,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = 'cpu',  # Note: Overridden by accelerator
        mixed_precision: bool = False,
        pre_train_processor: Optional[nn.Module] = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        gradient_accumulation_steps: int = 1,
        accelerator: Optional[Accelerator] = None,
        **kwargs
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.wandb_log = wandb_available and wandb_log and wandb.run is not None
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        
        self.accelerator = accelerator or Accelerator()
        self.device = self.accelerator.device
        self.mixed_precision = mixed_precision
        self.data_processor = pre_train_processor
        
        if int(gradient_accumulation_steps) < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.start_epoch = 0

    @property
    def _is_main(self) -> bool:
        return self.accelerator.is_main_process

    def _prepare_sample(self, sample: dict, step: Optional[int] = None) -> dict:
        """Helper to centralize data processing and device transfer."""
        if self.data_processor is not None:
            kwargs = {'step': step} if step is not None else {}
            return self.data_processor.preprocess(sample, **kwargs)
        return {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

    def _postprocess_output(self, out: torch.Tensor, sample: dict, step: Optional[int] = None):
        """Helper to centralize data post-processing."""
        if self.data_processor is not None:
            kwargs = {'step': step} if step is not None else {}
            return self.data_processor.postprocess(out, sample, **kwargs)
        return out, sample

    def train(
        self, train_loader, test_loaders, optimizer, scheduler,
        regularizer=None, training_loss=None, eval_losses=None, eval_modes=None,
        save_every: int=None, save_best: str=None,
        save_dir: Union[str, Path]="./ckpt", resume_from_dir: Union[str, Path]=None,
        max_autoregressive_steps: int=None,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer
        self.save_every = save_every
        self.save_best = save_best

        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        training_loss = training_loss or LpLoss(d=2)
        if getattr(training_loss, 'reduction', None) == "mean":
            warnings.warn(f"{training_loss.reduction=}. The Trainer expects losses to sum across the batch dim.")

        eval_losses = eval_losses or dict(l2=training_loss)
        eval_modes = eval_modes or {}
        
        self.model = self.model.to(self.device)
        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        if self.save_best:
            best_metric_value = float('inf')
            self.save_every = None  # Exclusive for simplicity
            expected_metrics = [f"{n}_{m}" for n in test_loaders for m in eval_losses]
            assert self.save_best in expected_metrics, f"Expected metric from {expected_metrics}, got {save_best}"

        if self.verbose and self._is_main:
            print(f'Training on {len(train_loader.dataset)} samples')
            print(f'Testing on {[len(l.dataset) for l in test_loaders.values()]} samples on resolutions {list(test_loaders.keys())}.')
            sys.stdout.flush()
        
        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(train_err=train_err, avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss, epoch_train_time=epoch_train_time)

            if self.wandb_log:
                self._wandb_log_model_parameters(epoch + 1)

            if epoch % self.eval_interval == 0:
                eval_metrics = self.evaluate_all(epoch, eval_losses, test_loaders, eval_modes, max_autoregressive_steps)
                epoch_metrics.update(eval_metrics)
                
                if self.save_best and eval_metrics[self.save_best] < best_metric_value:
                    best_metric_value = eval_metrics[self.save_best]
                    self.checkpoint(save_dir)

            if self.save_every and epoch % self.save_every == 0:
                self.checkpoint(save_dir)

            if self.wandb_log and self._is_main:
                wandb.log({}, step=epoch + 1, commit=True)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        self.epoch = epoch
        self.model.train()
        if self.data_processor: self.data_processor.train()
        
        avg_loss = avg_lasso_loss = train_err = 0.0
        self.n_samples = 0
        t1 = default_timer()

        self.optimizer.zero_grad(set_to_none=True)
        batch_iter = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{self.n_epochs}", leave=True, unit="batch") if self._is_main else train_loader

        grad_norms, last_grad_payload = [], {}

        for idx, sample in enumerate(batch_iter):
            with self.accelerator.accumulate(self.model):
                loss = self._compute_training_loss(idx, sample, training_loss)
                self.accelerator.backward(loss)

                # W&B Gradient metrics capture
                if self.accelerator.sync_gradients and self.wandb_log and self._is_main:
                    norm, payload = self._capture_grad_metrics()
                    grad_norms.append(norm)
                    last_grad_payload = payload

                if self.accelerator.sync_gradients:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        # Step Scheduler
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1
        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        avg_lasso_loss = (avg_lasso_loss / self.n_samples) if self.regularizer else None
        lr = self.optimizer.param_groups[0]["lr"]

        if self.verbose and (epoch % self.eval_interval == 0):
            self._print_training(epoch, epoch_train_time, avg_loss, train_err, avg_lasso_loss, lr)

        if self.wandb_log and self._is_main:
            self._log_epoch_to_wandb(epoch, train_err, epoch_train_time, avg_loss, avg_lasso_loss, lr, grad_norms, last_grad_payload)

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def _compute_training_loss(self, idx, sample, training_loss):
        if self.regularizer: self.regularizer.reset()
        
        sample = self._prepare_sample(sample)
        self.n_samples += sample["y"].numel() if isinstance(sample["y"], torch.Tensor) else 1

        with self.accelerator.autocast() if self.mixed_precision else contextlib.nullcontext():
            out = self.model(**sample["x"])
            if self.epoch == 0 and idx == 0 and self.verbose and self._is_main:
                print(f"Raw outputs shape: {out.shape}")
            
            out, sample = self._postprocess_output(out, sample)
            loss = training_loss(out, sample["y"])

        if self.regularizer:
            loss += self.regularizer.loss
        return loss

    def evaluate_all(self, epoch, eval_losses, test_loaders, eval_modes, max_autoregressive_steps=None):
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            mode = eval_modes.get(loader_name, "single_step")
            all_metrics.update(self.evaluate(eval_losses, loader, log_prefix=loader_name, mode=mode, max_steps=max_autoregressive_steps))
        
        if self.verbose or self.wandb_log:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None, mode="single_step", max_steps=None):
        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            self.data_processor = self.data_processor.to(self.device)
        
        self.model.eval()
        if self.data_processor: self.data_processor.eval()

        errors = {f"{log_prefix}_{k}": 0.0 for k in loss_dict.keys()}
        self.n_samples = 0

        batch_iter = tqdm(data_loader, desc=f"Eval {log_prefix or 'val'} ({mode})", leave=False, unit="batch") if self._is_main else data_loader

        with torch.no_grad():
            for sample in batch_iter:
                step_losses = self.eval_one_batch_autoreg(sample, loss_dict, max_steps) if mode == "autoregressive" else self.eval_one_batch(sample, loss_dict)
                for loss_name, val_loss in step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss
        
        return {k: v / self.n_samples for k, v in errors.items()}

    def eval_one_batch(self, sample: dict, eval_losses: dict):
        sample = self._prepare_sample(sample)
        self.n_samples += sample["y"].numel()
        
        out, sample = self._postprocess_output(self.model(**sample["x"]), sample)
        return {name: loss_fn(out, sample["y"]).item() for name, loss_fn in eval_losses.items()}
        
    def eval_one_batch_autoreg(self, sample: dict, eval_losses: dict, max_steps: int=None):
        step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}
        t = 0
        max_steps = max_steps or float('inf')
        sample_counted = False

        while sample is not None and t < max_steps:
            sample = self._prepare_sample(sample, step=t)
            if sample is None: break
            
            if not sample_counted:
                self.n_samples += sample["y"].numel()
                sample_counted = True

            out, sample = self._postprocess_output(self.model(**sample["x"]), sample, step=t)
            
            for loss_name, loss_fn in eval_losses.items():
                step_losses[loss_name] += loss_fn(out, sample["y"]).item()
            t += 1

        return {k: v / max(1, t) for k, v in step_losses.items()}

    # --- W&B and State Management Helpers --- #

    def _capture_grad_metrics(self):
        """Helper to compute W&B gradient payloads safely."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        total_norm_sq = 0.0
        payload = {}
        for pname, p in unwrapped.named_parameters():
            if p.grad is None: continue
            g = p.grad.detach().float()
            pnorm = g.norm(2).item()
            total_norm_sq += pnorm ** 2
            key = pname.replace(".", "/")
            payload[f"gradients/{key}/norm"] = pnorm
            g_flat = g.cpu().reshape(-1).numpy()
            if g_flat.max() != g_flat.min():
                payload[f"gradients/{key}/hist"] = wandb.Histogram(g_flat)
        
        total_norm = total_norm_sq ** 0.5
        payload["train/grad_norm"] = total_norm
        return total_norm, payload

    def _log_epoch_to_wandb(self, epoch, train_err, epoch_time, avg_loss, avg_lasso, lr, grad_norms, last_grad_payload):
        train_payload = {
            "train/train_err": train_err,
            "train/epoch_time_s": epoch_time,
            "train/avg_loss": avg_loss,
            "train/lr": lr,
            "train/samples_per_sec": self.n_samples / epoch_time if epoch_time > 0 else 0,
        }
        if avg_lasso is not None: train_payload["train/avg_lasso_loss"] = avg_lasso
        if grad_norms:
            train_payload["train/grad_norm_mean"] = sum(grad_norms) / len(grad_norms)
            train_payload["train/grad_norm_max"] = max(grad_norms)
        if torch.cuda.is_available():
            train_payload["system/gpu_memory_allocated_GB"] = torch.cuda.memory_allocated() / 1e9
            train_payload["system/gpu_memory_reserved_GB"] = torch.cuda.memory_reserved() / 1e9
            
        wandb.log({k: _wandb_numeric(v) for k, v in train_payload.items() if v is not None}, step=epoch + 1, commit=False)
        if last_grad_payload:
            wandb.log(last_grad_payload, step=epoch + 1, commit=False)

    def _print_training(self, epoch, time, avg_loss, train_err, avg_lasso_loss=None, lr=None):
        if self._is_main:
            parts = [f"[{epoch}] time={time:.2f}", f"avg_loss={avg_loss:.4f}", f"train_err={train_err:.4f}"]
            if avg_lasso_loss is not None: parts.append(f"avg_lasso={avg_lasso_loss:.4f}")
            if lr is not None: parts.append(f"lr={lr:g}")
            print(", ".join(parts))
            sys.stdout.flush()
    
    def log_eval(self, epoch: int, eval_metrics: dict):
        parts = [(m, _wandb_numeric(v)) for m, v in eval_metrics.items() if isinstance(v, (int, float, torch.Tensor))]
        if self.verbose and parts and self._is_main:
            print("Eval: " + ", ".join(f"{m}={v:.4f}" for m, v in parts))
            sys.stdout.flush()

        if self.wandb_log and parts and self._is_main:
            wandb.log({f"eval/{m}": v for m, v in parts}, step=epoch + 1, commit=False)

    def _wandb_log_model_parameters(self, step: int):
        if not (self.wandb_log and self._is_main): return
        payload = {}
        with torch.no_grad():
            total_param_norm_sq = 0.0
            for name, p in self.accelerator.unwrap_model(self.model).named_parameters():
                key = name.replace(".", "/")
                hist = _wandb_histogram_param(p)
                if hist is not None:
                    payload[f"params/{key}/hist"] = hist
                    pnorm = p.detach().float().norm(2).item()
                    payload[f"params/{key}/norm"] = pnorm
                    total_param_norm_sq += pnorm ** 2
            payload["params/global_weight_norm"] = total_param_norm_sq ** 0.5
        if payload: wandb.log(payload, step=step, commit=False)

    def resume_state_from_dir(self, save_dir):
        save_dir = Path(save_dir)
        save_name = "best_model" if (save_dir / "best_model_state_dict.pt").exists() else "model"
        if not (save_dir / f"{save_name}_state_dict.pt").exists():
            raise FileNotFoundError("Error: resume_from_dir expects model.pt or best_model.pt.")
            
        _, self.optimizer, self.scheduler, self.regularizer, resume_epoch = load_training_state(
            save_dir=save_dir, save_name=save_name,
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer, regularizer=self.regularizer, scheduler=self.scheduler
        )

        if resume_epoch and resume_epoch > self.start_epoch:
            self.start_epoch = resume_epoch
            if self.verbose and self._is_main: print(f"Trainer resuming from epoch {resume_epoch}")

    def checkpoint(self, save_dir):
        if self._is_main:
            save_training_state(
                save_dir=save_dir, save_name='best_model' if self.save_best else "model",
                model=self.accelerator.unwrap_model(self.model),
                optimizer=self.optimizer, scheduler=self.scheduler, regularizer=self.regularizer, epoch=self.epoch
            )
            if self.verbose: print(f"[Rank 0]: saved training state to {save_dir}")
        self.accelerator.wait_for_everyone()
