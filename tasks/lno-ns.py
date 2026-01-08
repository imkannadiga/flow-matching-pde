from tasks.base import BaseTask
import torch
from neuralop.training.trainer import Trainer
from pathlib import Path


class LNONSTask(BaseTask):
    def run(self):
        # Load data
        train_loader, test_loader = self.dataset.get_dataloaders()

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.lr)
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.train_cfg.scheduler_step_size, gamma=self.train_cfg.scheduler_gamma)
        
        # Training configuration
        trainer = Trainer(
            model=self.model,
            n_epochs=self.train_cfg.epochs,
            wandb_log=self.wandb_cfg.use_wandb,
            device=torch.device(self.train_cfg.device),
            eval_interval=self.train_cfg.eval_int,
            verbose=self.train_cfg.verbose,
        )
        
        test_loaders = {
            "test": test_loader,
        }
        
        train_metrics = trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            save_every=self.train_cfg.save_int,
            save_dir=self.train_cfg.save_path,
        )
        
        # Log training metrics to local save directory
        print(f"[INFO] Training metrics: {train_metrics}")
        
        # Save final metrics
        metrics_path = Path(self.train_cfg.save_path) / "training_metrics.pt"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_metrics, metrics_path)
        print(f"[INFO] Training metrics saved to {metrics_path}")
        
        print(f"[INFO] Training completed for model: {self.model.__class__.__name__}")
        