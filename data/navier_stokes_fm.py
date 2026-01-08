from data.base import BaseDataModule, BaseSequentialDataset
import torch
from torch.utils.data import Dataset, DataLoader


class FlowMatchingSequentialDataset(Dataset):
    """
    Dataset that stores preprocessed (X, Y, T) time-step pairs for flow matching.
    Includes time information for each sample.
    """
    def __init__(self, data_tensor: torch.Tensor):
        # data_tensor: [N, T, C, H, W]
        self.X, self.Y = self.create_sequential_pairs(data_tensor)
        # Create time indices for each sample (normalized to [0, 1))
        # For each trajectory, we have T-1 pairs, each corresponding to a time step
        n_pairs = self.X.shape[0]
        n_trajectories = data_tensor.shape[0]
        time_steps_per_traj = data_tensor.shape[1] - 1
        self.T = torch.zeros(n_pairs)
        
        # Assign time steps (normalized to [0, 1) for flow matching)
        for i in range(n_trajectories):
            start_idx = i * time_steps_per_traj
            end_idx = start_idx + time_steps_per_traj
            # Normalize time steps to [0, 1) range
            self.T[start_idx:end_idx] = torch.linspace(0, 1 - 1/time_steps_per_traj, time_steps_per_traj)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.Y[idx],
            "t": self.T[idx]
        }

    @staticmethod
    def create_sequential_pairs(data):
        """
        Converts [N, T, C, H, W] → X, Y pairs.
        X = data at t, Y = data at t+1
        """
        N, T, C, H, W = data.shape
        X = data[:, :-1]
        Y = data[:, 1:]

        X = X.reshape(-1, C, H, W)
        Y = Y.reshape(-1, C, H, W)

        return X, Y


class NSFMDataModule(BaseDataModule):
    """
    Navier-Stokes data module for flow matching tasks.
    Extends BaseDataModule to provide time information for flow matching.
    """
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        Expects [T, H, W, N] → returns [N, T, C, H, W]
        Same preprocessing as NSDataModule but uses FlowMatchingSequentialDataset.
        """
        data = data.permute(3, 0, 1, 2)      # [N, T, H, W]
        data = data.unsqueeze(2)             # [N, T, 1, H, W]
        return data
    
    def get_dataloaders(self):
        """
        Loads, processes, splits the dataset into train/test and returns dataloaders.
        Uses FlowMatchingSequentialDataset to include time information.
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = FlowMatchingSequentialDataset(processed)

        # Perform splits
        torch.manual_seed(self.seed)
        n_total = len(dataset)
        n_tr = self.cfg.n_tr
        n_te = self.cfg.n_te
        splits = [n_tr, n_te, n_total - n_tr - n_te]
        data_tr, data_te, _ = torch.utils.data.random_split(dataset, splits)

        # Dataloaders
        train_loader = DataLoader(data_tr, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(data_te, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    
    def get_testing_data(self, n_samples: int = None, data_loader: bool = True):
        """
        Returns testing data for evaluation.
        If n_samples is None, returns all test data.
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = FlowMatchingSequentialDataset(processed)
        
        # Get test split
        torch.manual_seed(self.seed)
        n_total = len(dataset)
        n_tr = self.cfg.n_tr
        n_te = self.cfg.n_te if n_samples is None else n_samples
        splits = [n_tr, n_te, n_total - n_tr - n_te]
        _, test_dataset, _ = torch.utils.data.random_split(dataset, splits)
        
        # If n_samples is specified and less than available, take subset
        if n_samples is not None and n_samples < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:n_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices.tolist())
        
        if data_loader:
            return test_dataset, DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_dataset, test_dataset
