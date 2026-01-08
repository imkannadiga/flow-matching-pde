from tasks.base import BaseTask
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

# Import Flow Matching utilities if available
try:
    from util.gaussian_process import GPPrior
    from util.util import make_grid, reshape_for_batchwise
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("Warning: GPPrior not available. Using simple Gaussian noise for Flow Matching.")


def flow_matching_loss(v_pred, v_target):
    """
    Flow Matching loss: MSE between predicted and target velocity fields.
    
    This implements the Flow Matching loss from Lipman et al. (2023):
    L = ||v_θ(x_t, t) - u_t(x_t | x_1)||^2
    
    where:
    - v_θ(x_t, t) is the model's predicted velocity
    - u_t(x_t | x_1) is the conditional velocity field from the probability path
    """
    return F.mse_loss(v_pred, v_target)


class FMNSTask(BaseTask):
    """
    Flow Matching task for Navier-Stokes equations.
    
    Implements Flow Matching as described in:
    Lipman et al., "Flow Matching for Generative Modeling", 2023
    
    Key components:
    1. Random time sampling: t ~ U(0,1)
    2. Conditional probability path: p_t(x | x_1) where x_1 is a data sample
    3. Conditional velocity field: u_t(x | x_1) = d/dt x_t
    4. Training objective: E[||v_θ(x_t, t) - u_t(x_t | x_1)||^2]
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Flow Matching parameters
        self.sigma_min = getattr(self.train_cfg, 'sigma_min', 1e-4)
        self.use_vp = getattr(self.train_cfg, 'use_vp', False)
        
        # Initialize GP prior if available
        if GP_AVAILABLE:
            kernel_length = getattr(self.train_cfg, 'kernel_length', 0.001)
            kernel_variance = getattr(self.train_cfg, 'kernel_variance', 1.0)
            self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device=self.train_cfg.device)
        else:
            self.gp = None
        
        if self.use_vp:
            self.alpha, self.dalpha = self._construct_alpha()
    
    def _construct_alpha(self):
        """Construct alpha schedule for variance preserving (VP) flow."""
        device = self.train_cfg.device if isinstance(self.train_cfg.device, torch.device) else torch.device(self.train_cfg.device)
        
        def alpha(t):
            return torch.cos((t + 0.08) / 2.16 * np.pi).to(device)
        
        def dalpha(t):
            return -np.pi / 2.16 * torch.sin((t + 0.08) / 2.16 * np.pi).to(device)
        
        return alpha, dalpha
    
    def _sample_noise(self, batch_size, shape, device):
        """Sample noise for the probability path."""
        if self.gp is not None:
            # Use GP prior for structured noise
            dims = shape[2:]  # Spatial dimensions
            query_points = make_grid(dims)
            noise = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=shape[1])
            return noise
        else:
            # Fallback to simple Gaussian noise
            return torch.randn(batch_size, *shape[1:], device=device)
    
    def _sample_from_path(self, t, x_data):
        """
        Sample from conditional probability path p_t(x | x_data).
        
        For non-VP (linear interpolation path):
        - mu_t = t * x_data
        - sigma_t = 1 - (1 - sigma_min) * t
        - x_t ~ N(mu_t, sigma_t^2 I)
        
        For VP (variance preserving):
        - alpha_t = cos((t + 0.08)/2.16 * pi)
        - mu_t = alpha_t * x_data
        - sigma_t = sqrt(1 - alpha_t^2)
        """
        batch_size = x_data.shape[0]
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        # Sample noise
        noise = self._sample_noise(batch_size, x_data.shape, x_data.device)
        
        # Reshape t for broadcasting: [batch_size] -> [batch_size, 1, 1, ...] (matching x_data dims)
        if GP_AVAILABLE:
            t_expanded = reshape_for_batchwise(t, 1 + n_dims)
        else:
            # Manual reshape: [batch_size] -> [batch_size, 1, 1, 1, ...]
            t_expanded = t.view(-1, *([1] * (1 + n_dims)))
        
        if self.use_vp:
            # Variance preserving path
            mu = self.alpha(1 - t_expanded) * x_data
            sigma = torch.sqrt(1 - self.alpha(1 - t_expanded) ** 2)
        else:
            # Linear interpolation path (default)
            mu = t_expanded * x_data
            sigma = 1.0 - (1.0 - self.sigma_min) * t_expanded
        
        # Sample from path
        x_t = mu + sigma * noise
        return x_t
    
    def _compute_conditional_velocity(self, t, x_data, x_t):
        """
        Compute conditional velocity field u_t(x_t | x_data).
        
        This is the derivative of the probability path with respect to time.
        
        For non-VP:
        u_t = (x_data - (1 - sigma_min) * x_t) / c
        where c = 1 - (1 - sigma_min) * t
        
        For VP:
        u_t = (dalpha/dt) / (1 - alpha^2) * (alpha * x_t - x_data)
        """
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        # Reshape t for broadcasting: [batch_size] -> [batch_size, 1, 1, ...] (matching x_data dims)
        if GP_AVAILABLE:
            t_expanded = reshape_for_batchwise(t, 1 + n_dims)
        else:
            # Manual reshape: [batch_size] -> [batch_size, 1, 1, 1, ...]
            t_expanded = t.view(-1, *([1] * (1 + n_dims)))
        
        if self.use_vp:
            # Variance preserving velocity field
            alpha_val = self.alpha(1 - t_expanded)
            dalpha_val = self.dalpha(1 - t_expanded)
            u_t = (dalpha_val / (1 - alpha_val ** 2)) * (alpha_val * x_t - x_data)
        else:
            # Linear interpolation velocity field
            c = 1.0 - (1.0 - self.sigma_min) * t_expanded
            u_t = (x_data - (1.0 - self.sigma_min) * x_t) / c
        
        return u_t
    
    def run(self):
        """
        Run the training loop for Flow Matching on the Navier-Stokes dataset.
        
        Implements the Flow Matching algorithm:
        1. Sample data x_1 ~ p_1 (from dataset)
        2. Sample time t ~ U(0,1)
        3. Sample x_t ~ p_t(x | x_1)
        4. Compute target velocity u_t = get_conditional_velocity(t, x_1, x_t)
        5. Train model: min ||v_θ(x_t, t) - u_t||^2
        """
        # Load data - use regular NSDataModule (not NSFMDataModule)
        # We'll handle Flow Matching setup ourselves
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

        checkpoint_dir = Path(self.train_cfg.save_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_metrics = []
        test_metrics = []

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_batches = 0

            for step, batch in enumerate(train_loader, 1):
                # Get data sample x_1 ~ p_1 (from dataset)
                # Both x and y are valid data samples, randomly choose one
                # This ensures we use all data samples for Flow Matching
                if step % 2 == 0:
                    x_data = batch["x"].to(device)
                else:
                    x_data = batch["y"].to(device)
                
                # Flow Matching setup:
                # 1. Sample random time t ~ U(0,1)
                batch_size = x_data.shape[0]
                t = torch.rand(batch_size, device=device)
                
                # 2. Sample from conditional path: x_t ~ p_t(x | x_data)
                x_t = self._sample_from_path(t, x_data)
                
                # 3. Compute conditional velocity field: u_t(x_t | x_data)
                u_target = self._compute_conditional_velocity(t, x_data, x_t)
                
                # 4. Model predicts velocity: v_θ(x_t, t)
                v_pred = self.model(t, x_t)
                
                # 5. Flow Matching loss: ||v_θ(x_t, t) - u_t||^2
                loss = flow_matching_loss(v_pred, u_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            avg_train_loss = total_loss / total_batches
            train_metrics.append(avg_train_loss)
            print(f"[Epoch {epoch}] Avg Training Loss: {avg_train_loss:.6f}")

            # Evaluation
            if epoch % eval_int == 0 or epoch == num_epochs:
                self.model.eval()
                total_test_loss = 0.0
                test_batches = 0

                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_loader):
                        # Get data sample x_1 ~ p_1 (from dataset)
                        if batch_idx % 2 == 0:
                            x_data = batch["x"].to(device)
                        else:
                            x_data = batch["y"].to(device)
                        
                        # Sample random time for evaluation
                        batch_size = x_data.shape[0]
                        t = torch.rand(batch_size, device=device)
                        
                        # Sample from path and compute target
                        x_t = self._sample_from_path(t, x_data)
                        u_target = self._compute_conditional_velocity(t, x_data, x_t)
                        
                        # Predict and compute loss
                        v_pred = self.model(t, x_t)
                        loss = flow_matching_loss(v_pred, u_target)

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
        checkpoint_path = checkpoint_dir / "manifest.pt"
        torch.save({
            "model": "model_state_dict.pt",
        }, checkpoint_path)
        torch.save(self.model.state_dict(), checkpoint_dir / "model_state_dict.pt")
        print(f"Final model saved at {checkpoint_path}")
        print(f"[INFO] Final Training Metrics: {train_metrics}")
        if test_metrics:
            print(f"[INFO] Final Test Metrics: {test_metrics}")
