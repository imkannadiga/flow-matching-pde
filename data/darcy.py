from data.base import BaseDataModule
import h5py
import torch

class DarcyDataModule(BaseDataModule):
    def __init__(self, data_path, beta=None, append_beta_channel=True):
        """
        Args:
            data_path: Path to a single Darcy HDF5 file.
            beta: The beta float value (e.g., 2.50). If None, defaults to 1-channel conditioning.
            append_beta_channel: If True and beta is provided, appends beta as a spatial map.
            transform: Optional torchvision/custom transforms.
        """
        super().__init__(data_path)
        
        # Load the raw arrays into memory (or keep as h5py references if RAM is limited)
        with h5py.File(data_path, 'r') as f:
            # Assumes shapes are (N, D, D) and (N, C, D, D) based on your original code
            self.nu = torch.tensor(f["nu"][:], dtype=torch.float32)
            self.pressure = torch.tensor(f["tensor"][:], dtype=torch.float32)
            
        # Ensure 'nu' has a channel dimension: [N, 1, D, D]
        if self.nu.dim() == 3:
            self.nu = self.nu.unsqueeze(1)
            
        self.beta = beta
        self.append_beta_channel = append_beta_channel

        # Dynamically set expected channel depths for the neural network
        self.c_channels = 2 if (self.beta is not None and self.append_beta_channel) else 1
        self.target_channels = self.pressure.shape[1] 

    def __len__(self):
        return self.nu.shape[0]

    def _fetch_data_pair(self, idx):
        # 1. Base conditioning: Permeability field [1, D, D]
        C = self.nu[idx]
        
        # 2. Target: Pressure field [C, D, D]
        X_target = self.pressure[idx]

        # 3. Concatenate Beta spatially if requested
        if self.beta is not None and self.append_beta_channel:
            # Create a constant map of the beta value matching the spatial dims
            beta_map = torch.full_like(C, self.beta)
            
            # C becomes [2, D, D]
            C = torch.cat([C, beta_map], dim=0)

        return C, X_target