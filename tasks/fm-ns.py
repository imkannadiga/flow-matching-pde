from tasks.base import BaseTask
import torch
from pathlib import Path


def flow_matching_loss(v_pred, x_t, x_tp, dt):
    """Compute flow matching loss as MSE between predicted and true velocity."""
    v_true = (x_tp - x_t) / dt
    return ((v_pred - v_true) ** 2).mean()


class FMNSTask(BaseTask):
    def run(self):
        """
        Run the training loop for flow matching on the Navier-Stokes dataset.
        Supports periodic evaluation and model checkpointing.
        """
        # Load data
        train_loader, test_loader = self.dataset.get_dataloaders()

        # Setup device
        device = torch.device(self.train_cfg.device)
        self.model.to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.train_cfg.scheduler_step_size,
            gamma=self.train_cfg.scheduler_gamma,
        )

        # Training settings
        num_epochs = self.train_cfg.epochs
        eval_int = self.train_cfg.eval_int
        save_int = self.train_cfg.save_int
        dt = 1.0 / 50.0  # Assuming fixed Δt

        checkpoint_dir = Path(self.train_cfg.save_path)
        checkpoint_dir.mkdir(parents=True)

        train_metrics = []
        test_metrics = []

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_batches = 0

            for step, batch in enumerate(train_loader, 1):
                x_t = batch["x"].to(device)
                x_tp = batch["y"].to(device)
                t = batch["t"].to(device)

                # Forward pass
                v_pred = self.model(x_t, t)
                loss = flow_matching_loss(v_pred, x_t, x_tp, dt)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

                # print(f"[Epoch {epoch} | Step {step}] Training Loss: {loss.item():.6f}")

            avg_train_loss = total_loss / total_batches
            train_metrics.append(avg_train_loss)
            print(f"[Epoch {epoch}] Avg Training Loss: {avg_train_loss:.6f}")

            # Evaluation
            if epoch % eval_int == 0 or epoch == num_epochs:
                self.model.eval()
                total_test_loss = 0.0
                test_batches = 0

                with torch.no_grad():
                    for batch in test_loader:
                        x_t = batch["x"].to(device)
                        x_tp = batch["y"].to(device)
                        t = batch["t"].to(device)

                        v_pred = self.model(x_t, t)
                        loss = flow_matching_loss(v_pred, x_t, x_tp, dt)

                        total_test_loss += loss.item()
                        test_batches += 1

                avg_test_loss = total_test_loss / test_batches
                test_metrics.append(avg_test_loss)
                print(f"[Epoch {epoch}] Avg Test Loss: {avg_test_loss:.6f}")

            # Save checkpoint
            if epoch % save_int == 0 or epoch == num_epochs:
                checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                }, checkpoint_path)
                print(f"[Epoch {epoch}] Checkpoint saved to {checkpoint_path}")

            scheduler.step()

        print("[INFO] Training complete.")
        checkpoint_path = checkpoint_dir / f"manifest.pt"
        torch.save({
            "model": self.model.state_dict()
        }, checkpoint_path)
        print(f"Final model saved at {checkpoint_path}")
        print(f"[INFO] Final Training Metrics: {train_metrics}")
        if test_metrics:
            print(f"[INFO] Final Test Metrics: {test_metrics}")
