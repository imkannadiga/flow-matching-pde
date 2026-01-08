import os
import abc
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig


class BaseSequentialDataset(Dataset):
    """
    Dataset that stores preprocessed (X, Y) time-step pairs.
    """
    def __init__(self, data_tensor: torch.Tensor):
        # data_tensor: [N, T, C, H, W]
        self.X, self.Y = self.create_sequential_pairs(data_tensor)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"x":self.X[idx], "y":self.Y[idx]}

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


class BaseDataModule(abc.ABC):
    def __init__(self, name: str, data_dir: str, filename: str, batch_size: int, n_tr:int, n_te:int, **kwargs):
        self.name = name
        self.cfg = DictConfig({
            'root_dir': data_dir,
            'filename': filename,
            'n_tr': n_tr,
            'n_te': n_te
        })
        self.batch_size = batch_size
        self.seed = 42
        
    def get_dataloaders(self):
        """
        Loads, processes, splits the dataset into train/test and returns dataloaders.
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = BaseSequentialDataset(processed)

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

    def get_testing_data(self, n_samples=None, data_loader=True):
        """
        Returns testing data for evaluation.
        
        Args:
            n_samples: Number of samples to extract. If None, uses n_te from config.
            data_loader: Whether to return a DataLoader or Dataset.
        
        Returns:
            If data_loader=True: (dataset, DataLoader)
            If data_loader=False: dataset
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = BaseSequentialDataset(processed)
        
        # Determine number of test samples
        if n_samples is None:
            n_samples = self.cfg.n_te
        
        # Split dataset
        torch.manual_seed(self.seed)
        n_total = len(dataset)
        n_tr = self.cfg.n_tr
        n_te = min(n_samples, n_total - n_tr)
        splits = [n_tr, n_te, n_total - n_tr - n_te]
        _, test_dataset, _ = random_split(dataset, splits)
        
        # If n_samples is specified and less than available, take subset
        if n_samples is not None and n_samples < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:n_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices.tolist())

        if data_loader:
            return test_dataset, DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_dataset

    def read_data(self):
        """
        Reads raw data file. Override for dataset-specific loading logic.
        """
        file_path = os.path.join(self.cfg.root_dir, self.cfg.filename)
        if file_path.endswith(".npy"):
            data = torch.from_numpy(np.load(file_path, allow_pickle=True)).float()
        elif file_path.endswith(".pt"):
            data = torch.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return data

    @abc.abstractmethod
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        Override to apply dataset-specific transformations. Should return shape [N, T, C, H, W].
        """
        pass
