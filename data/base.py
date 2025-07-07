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
        return {"x":self.X[idx],"y":self.Y[idx]}

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
    def __init__(self, name: str, data_dir: str, filename: str, batch_size: int, n_tr:int, n_te:int):
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

    def get_testing_data(self):
        """
        Returns raw tensor and testing dataloader.
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        dataset = BaseSequentialDataset(processed)

        torch.manual_seed(self.seed)
        splits = [self.cfg.n_tr, self.cfg.n_te, len(dataset) - self.cfg.n_tr - self.cfg.n_te]
        test_split = random_split(dataset, splits)[1]

        return processed, DataLoader(test_split, batch_size=self.batch_size, shuffle=False)

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
