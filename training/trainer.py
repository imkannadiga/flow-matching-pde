from timeit import default_timer
from pathlib import Path
from typing import Any, Union
import sys
import warnings

import torch
from tqdm import tqdm
from torch import nn
from accelerate import Accelerator
# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

from neuralop.losses import LpLoss
from .training_state import load_training_state, save_training_state


def _wandb_numeric(x: Any) -> float:
    """Convert torch / numpy scalars to a Python float for W&B (charts need CPU scalars)."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if hasattr(x, "item") and callable(x.item) and not isinstance(x, (float, int, bool)):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


def _wandb_histogram_param(tensor: torch.Tensor):
    """Build a W&B histogram from a parameter tensor (CPU, detached)."""
    flat = tensor.detach().cpu().float().reshape(-1)
    if flat.numel() == 0:
        return None
    return wandb.Histogram(flat.numpy())


class Trainer:
    """
    A general Trainer class to train models on given datasets. 

    .. note ::
        Our Trainer expects datasets to provide batches as key-value dictionaries, ex.: 
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses. 
        For specifics and an example, check ``neuralop.data.datasets.DarcyDataset``. 

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str 'cpu' or 'cuda'
    mixed_precision : bool, default is False
        whether to use torch.autocast to compute mixed precision
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use DDP
    verbose : bool, default is False
    gradient_accumulation_steps : int, default is 1
        Number of dataloader batches to accumulate before ``optimizer.step()`` (1 = disabled).
        Lets you approximate a larger batch size without raising peak activation memory: use a
        smaller per-step ``batch_size`` and set this to ``k`` for an effective update batch of
        ``k * batch_size`` (with sum-reduction losses, gradients match one pass over those samples).
    """
    def __init__(
        self,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool=False,
        device: str='cpu',
        mixed_precision: bool=False,
        pre_train_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,
        gradient_accumulation_steps: int = 1,
        accelerator: Accelerator=None,
        **kwargs
    ):
        """
        """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = (wandb_log and wandb.run is not None)
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        self.device = self.accelerator.device
        self.mixed_precision = mixed_precision
        self.data_processor = pre_train_processor
        ga = int(gradient_accumulation_steps)
        if ga < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        self.gradient_accumulation_steps = ga
    
        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def _progress_bar_enabled(self) -> bool:
        """Use tqdm only on the global main process."""
        return self.accelerator.is_main_process

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        eval_modes=None,
        save_every: int=None,
        save_best: int=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
        max_autoregressive_steps: int=None,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here. 

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        eval_modes: dict[str], optional
            optional mapping from the name of each loader to its evaluation mode.

            * if 'single_step', predicts one input-output pair and evaluates loss.
            
            * if 'autoregressive', autoregressively predicts output using last step's 
            output as input for a number of steps defined by the temporal dimension of the batch.
            This requires specially batched data with a data processor whose ``.preprocess`` and 
            ``.postprocess`` both take ``idx`` as an argument.
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        max_autoregressive_steps : int, default None
            if provided, and a dataloader is to be evaluated in autoregressive mode,
            limits the number of autoregressive in each rollout to be performed.
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)
        
        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction'):
            if training_loss.reduction == "mean":
                warnings.warn(f"{training_loss.reduction=}. This means that the loss is "
                              "initialized to average across the batch dim. The Trainer "
                              "expects losses to sum across the batch dim.")

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # create default eval modes
        if eval_modes is None:
            eval_modes = {}

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert self.save_best in metrics,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            if self.accelerator.is_main_process:
                print(f'Training on {len(train_loader.dataset)} samples')
                print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                      f'         on resolutions {[name for name in test_loaders]}.')
                sys.stdout.flush()
        
        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time
            )
            
            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                eval_losses=eval_losses,
                                                test_loaders=test_loaders,
                                                eval_modes=eval_modes,
                                                max_autoregressive_steps=max_autoregressive_steps)
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

            if self.wandb_log:
                self._wandb_log_model_parameters(epoch + 1)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        
        # track number of training examples in batch
        self.n_samples = 0

        self.optimizer.zero_grad(set_to_none=True)

        batch_iter = train_loader
        if self._progress_bar_enabled():
            batch_iter = tqdm(
                train_loader,
                desc=f"Train epoch {epoch + 1}/{self.n_epochs}",
                leave=True,
                unit="batch",
            )

        for idx, sample in enumerate(batch_iter):
            with self.accelerator.accumulate(self.model):
                loss = self._compute_training_loss(idx, sample, training_loss)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        will_eval = epoch % self.eval_interval == 0
        if self.verbose and will_eval:
            self._print_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )
        # W&B: independent of verbose (otherwise quiet runs only show system charts).
        if self.wandb_log and self.accelerator.is_main_process:
            train_payload = {
                "train/train_err": train_err,
                "train/epoch_time_s": epoch_train_time,
                "train/avg_loss": avg_loss,
                "train/lr": lr,
            }
            if avg_lasso_loss is not None:
                train_payload["train/avg_lasso_loss"] = avg_lasso_loss
            wandb.log(
                {
                    k: _wandb_numeric(v)
                    for k, v in train_payload.items()
                    if v is not None
                },
                step=epoch + 1,
                commit=not will_eval,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, epoch, eval_losses, test_loaders, eval_modes, max_autoregressive_steps=None):
        """evaluate_all iterates through the entire dict of test_loaders
        to perform evaluation on the whole dataset stored in each one. 

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_losses : dict[Loss]
            keyed ``loss_name: loss_obj`` for each pair. Full set of 
            losses to use in evaluation for each test loader. 
        test_loaders : dict[DataLoader]
            keyed ``loader_name: loader`` for each test loader. 
        eval_modes : dict[str], optional
            keyed ``loader_name: eval_mode`` for each test loader.
            * If ``eval_modes.get(loader_name)`` does not return a value, 
            the evaluation is automatically performed in ``single_step`` mode. 
        max_autoregressive_steps : ``int``, optional
            if provided, and one of the test loaders has ``eval_mode == "autoregressive"``,
            limits the number of autoregressive steps performed per rollout.

        Returns
        -------
        all_metrics: dict
            collected eval metrics for each loader. 
        """
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self.evaluate(eval_losses, loader,
                                    log_prefix=loader_name,
                                    mode=loader_eval_mode,
                                    max_steps=max_autoregressive_steps)   
            all_metrics.update(**loader_metrics)
        if self.verbose or self.wandb_log:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None, mode="single_step", max_steps=None):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        mode : Literal {'single_step', 'autoregressive'}
            if 'single_step', performs standard evaluation
            if 'autoregressive' loops through `max_steps` steps
        max_steps : int, optional
            max number of steps for autoregressive rollout. 
            If None, runs the full rollout.
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # Ensure model and data processor are loaded to the proper device

        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            self.data_processor = self.data_processor.to(self.device)
        
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, 'reduction'):
                if eval_loss.reduction == "mean":
                    warnings.warn(f"{eval_loss.reduction=}. This means that the loss is "
                                "initialized to average across the batch dim. The Trainer "
                                "expects losses to sum across the batch dim.")

        self.n_samples = 0
        with torch.no_grad():
            batch_iter = data_loader
            if self._progress_bar_enabled():
                batch_iter = tqdm(
                    data_loader,
                    desc=f"Eval {log_prefix or 'val'} ({mode})",
                    leave=False,
                    unit="batch",
                )
            for idx, sample in enumerate(batch_iter):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                if mode == "single_step":
                    eval_step_losses, outs = self.eval_one_batch(sample, loss_dict, return_output=return_output)
                elif mode == "autoregressive":
                    eval_step_losses, outs = self.eval_one_batch_autoreg(sample, loss_dict,
                                                                         return_output=return_output,
                                                                         max_steps=max_steps)

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss
        
        for key in errors.keys():
            errors[key] /= self.n_samples

        # on last batch, log model outputs
        # if self.log_output and self.wandb_log:
        #     errors[f"{log_prefix}_outputs"] = wandb.Image(outs)
        
        return errors
    
    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def _compute_training_loss(self, idx, sample, training_loss):
        """Forward + loss for one dataloader batch (no backward, no zero_grad, no optimizer step)."""
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        if self.mixed_precision:
            with self.accelerator.autocast():
                out = self.model(**sample["x"])
        else:
            out = self.model(**sample["x"])

        if self.epoch == 0 and idx == 0 and self.verbose and self.accelerator.is_main_process and isinstance(out, torch.Tensor):
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        if self.mixed_precision:
            with self.accelerator.autocast():
                loss = training_loss(out, sample["y"])
        else:
            loss = training_loss(out, sample["y"])

        if self.regularizer:
            loss = loss + self.regularizer.loss

        return loss

    def train_one_batch(self, idx, sample, training_loss):
        """Single optimizer update from one batch (full forward, backward, step)."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._compute_training_loss(idx, sample, training_loss)
        self.accelerator.backward(loss)
        self.optimizer.step()
        return loss
    
    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].size(0)

        out = self.model(**sample["x"])

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, sample["y"])
            eval_step_losses[loss_name] = val_loss
        
        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
        
    def eval_one_batch_autoreg(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False,
                       max_steps: int=None):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        max_steps: int
            number of timesteps to roll out
            typically the full trajectory length
            If max_steps is none, runs until the full length

            .. note::
                If a value for ``max_steps`` is not provided, a data_processor
                must be provided to handle rollout logic. 
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs


        """
        eval_step_losses = {loss_name: 0. for loss_name in eval_losses.keys()}
        # eval_rollout_losses = {loss_name: 0. for loss_name in eval_losses.keys()}

        t = 0
        if max_steps is None:
            max_steps = float('inf')

        # only increment the sample count once
        sample_count_incr = False

        while sample is not None and t < max_steps:
            
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                # load data to device if no preprocessor exists
                sample = {
                    k: v.to(self.device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }

            if sample is None:
                break
            
            # only increment the sample count once
            if not sample_count_incr:
                self.n_samples += sample["y"].shape[0]
                sample_count_incr = True

            out = self.model(**sample["x"])
                
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)
            
            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, sample["y"])
                eval_step_losses[loss_name] += step_loss
            
            t += 1
        # average over all steps of the final rollout
        for loss_name in eval_step_losses.keys():
            eval_step_losses[loss_name] /= t 

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
    
    def _print_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: float = None,
        lr: float = None,
    ):
        """Stdout-only training line (eval-interval gated by caller)."""
        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        if lr is not None:
            msg += f", lr={lr:g}"
        if self.accelerator.is_main_process:
            print(msg)
            sys.stdout.flush()

    def log_training(self, 
            epoch:int,
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            lr: float=None
            ):
        """Legacy hook: console only. W&B train metrics are logged from ``train_one_epoch``."""
        self._print_training(
            epoch=epoch,
            time=time,
            avg_loss=avg_loss,
            train_err=train_err,
            avg_lasso_loss=avg_lasso_loss,
            lr=lr,
        )
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        parts = []
        for metric, value in eval_metrics.items():
            try:
                v = _wandb_numeric(value)
            except (TypeError, ValueError):
                continue
            parts.append((metric, v))

        if self.verbose and parts and self.accelerator.is_main_process:
            msg = "Eval: " + ", ".join(f"{m}={v:.4f}" for m, v in parts)
            print(msg)
            sys.stdout.flush()

        if self.wandb_log and parts and self.accelerator.is_main_process:
            wandb.log(
                {f"eval/{m}": v for m, v in parts},
                step=epoch + 1,
                commit=True,
            )

    def _wandb_log_model_parameters(self, step: int) -> None:
        """Log W&B histograms for every model parameter once per epoch (rank 0 only).

        Called from ``train()`` at the end of each epoch, not from the evaluation loop.
        """
        if not self.wandb_log:
            return
        if not self.accelerator.is_main_process:
            return
        model = self.accelerator.unwrap_model(self.model)
        payload = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                hist = _wandb_histogram_param(p)
                if hist is not None:
                    payload[f"params/{name.replace('.', '/')}"] = hist
        if payload:
            wandb.log(payload, step=step, commit=True)

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`
        
        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError("Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt.")
        # Load into the unwrapped module to keep accelerator wrapping intact.
        model_to_load = self.accelerator.unwrap_model(self.model)
        _, self.optimizer, self.scheduler, self.regularizer, resume_epoch =\
            load_training_state(save_dir=save_dir, save_name=save_name,
                                model=model_to_load,
                                optimizer=self.optimizer,
                                regularizer=self.regularizer,
                                scheduler=self.scheduler)

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose and self.accelerator.is_main_process:
                    print(f"Trainer resuming from epoch {resume_epoch}")


    def checkpoint(self, save_dir):
        """checkpoint saves current training state
        to a directory for resuming later. Only saves 
        training state on the first GPU. 
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if self.accelerator.is_main_process:
            if self.save_best is not None:
                save_name = 'best_model'
            else:
                save_name = "model"
            model_to_save = self.accelerator.unwrap_model(self.model)
            save_training_state(save_dir=save_dir, 
                                save_name=save_name,
                                model=model_to_save,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                regularizer=self.regularizer,
                                epoch=self.epoch
                                )
            if self.verbose:
                print(f"[Rank 0]: saved training state to {save_dir}")
        self.accelerator.wait_for_everyone()

       