from data.base import BaseDataModule, BaseSequentialDataset
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

class FMSequentialDataset(Dataset):
    """
    Dataset that stores preprocessed (X, Y) time-step pairs.
    """
    def __init__(self, data_tensor: torch.Tensor):
        # data_tensor: [N, T, C, H, W]
        self.X, self.Y, self.T = self.create_sequential_pairs(data_tensor)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.Y[idx], "t": self.T[idx]}

    @staticmethod
    def create_sequential_pairs(data):
        """
        Converts [N, T, C, H, W] → X, Y, sample_indices, time_steps.
        X = data at t, Y = data at t+1
        sample_indices = which sample each pair came from (0 to N-1)
        time_steps = time t of each pair
        """
        N, T, C, H, W = data.shape

        X = data[:, :-1]  # shape: [N, T-1, C, H, W]
        Y = data[:, 1:]   # shape: [N, T-1, C, H, W]

        # Flatten across N and T-1
        X = X.reshape(-1, C, H, W)  # shape: [N*(T-1), C, H, W]
        Y = Y.reshape(-1, C, H, W)

        # Create time steps
        time_steps = torch.arange(T - 1).repeat(N)                 # [0,1,...,T-2, 0,1,...]

        return X, Y, time_steps

class NSFMDataModule(BaseDataModule):
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        Expects [T, H, W, N] → returns [N, T, C, H, W]
        """
        data = data.permute(3, 0, 1, 2)      # [N, T, H, W]
        data = data.unsqueeze(2)             # [N, T, 1, H, W]
        return data
    
    def get_dataloaders(self):
        """
        Loads, processes, splits the dataset into train/test and returns dataloaders.
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = FMSequentialDataset(processed)

        # Perform splits
        torch.manual_seed(self.seed)
        n_total = len(dataset)
        n_tr = self.cfg.n_tr
        n_te = self.cfg.n_te
        splits = [n_tr, n_te, n_total - n_tr - n_te]
        data_tr, data_te, _ = random_split(dataset, splits)

        # Dataloaders
        train_loader = DataLoader(data_tr, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(data_te, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    
    
    def get_testing_data(self):
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = FMSequentialDataset(processed)

        torch.manual_seed(self.seed)
        splits = [self.cfg.n_tr, self.cfg.n_te, len(dataset) - self.cfg.n_tr - self.cfg.n_te]
        test_split = random_split(dataset, splits)[1]

        return processed, DataLoader(test_split, batch_size=self.batch_size, shuffle=False)
