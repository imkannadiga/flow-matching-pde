import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from omegaconf import DictConfig

class NavierStokesDataset(Dataset):
    def __init__(self, data_tensor):
        """
        data_tensor: torch.Tensor of shape [N, T, C, H, W]
        """
        self.data = data_tensor
        N, T, C, H, W = data_tensor.shape

        self.X = data_tensor[:, :-1]  # shape: [N, T-1, C, H, W]
        self.Y = data_tensor[:, 1:]   # shape: [N, T-1, C, H, W]

        self.X = self.X.reshape(-1, C, H, W)  # [N*(T-1), C, H, W]
        self.Y = self.Y.reshape(-1, C, H, W)  # [N*(T-1), C, H, W]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_data(cfg: DictConfig):
    '''
    Reads data file name from task specific config, reads the data, pre-process it
    Returns the training and testing data loader of batch size
    '''
    file_path = os.path.join(cfg.data.root_dir, cfg.data.filename)
    
    dataset = read_data(file_path)
    
    # Rearrange to [N, T, C, H, W]
    dataset = dataset.permute(3, 0, 1, 2)   # [N, T, H, W]
    dataset = dataset.unsqueeze(2)         # Add channel dim → [N, T, 1, H, W]

    # Wrap with custom Dataset BEFORE split
    full_dataset = NavierStokesDataset(dataset)
    data_tr, data_te, _ = random_split(full_dataset, [cfg.data.n_tr, cfg.data.n_te, len(full_dataset) - cfg.data.n_tr - cfg.data.n_te])
    
    train_loader = DataLoader(data_tr, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(data_te, batch_size=cfg.data.batch_size, shuffle=False)

    return train_loader, test_loader


def load_testing_data(cfg: DictConfig):
    '''
    Loads data and reshapes it into a dataset of shape [N, T, C, H, W] for testing
    '''

    # Load file
    file_path = os.path.join(cfg.data.root_dir, cfg.data.filename)
    dataset = read_data(file_path)  # [T, H, W, N]
    
    # Rearrange to [N, T, C, H, W]
    dataset = dataset.permute(3, 0, 1, 2)   # [N, T, H, W]
    raw_dataset = dataset.unsqueeze(2)         # Add channel dim → [N, T, 1, H, W]

    dataset = NavierStokesDataset(raw_dataset)
    dataset = random_split(dataset, [cfg.data.n_tr, cfg.data.n_te, len(dataset) - cfg.data.n_tr - cfg.data.n_te])[1]  # Get only test data

    seq_pairs = NavierStokesDataset(dataset)

    return raw_dataset, DataLoader(seq_pairs)


def read_data(file_path):
    '''
    data specific read logic
    '''
    if file_path.endswith('.npy'):
        data = torch.from_numpy(np.load(file_path, allow_pickle=True))
        return torch.tensor(data, dtype=torch.float32)
    elif file_path.endswith('.pt'):
        return torch.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess(data):
    '''
    data pre-processing pipeline (if needed)
    '''
    data = data.permute(3, 1, 2, 0)
    data = data.permute(0, -1, 1, 2).contiguous().reshape(-1, 64, 64).unsqueeze(1) 
    return data

def create_sequential_pairs(data):
    """
    Given a tensor of shape [N, T, C, H, W], return X and Y:
    - X: [N*(T-1), C, H, W] at time t
    - Y: [N*(T-1), C, H, W] at time t+1
    """
    N, T, C, H, W = data.shape

    # Get all t and t+1 pairs
    X = data[:, :-1]  # shape: [N, T-1, C, H, W]
    Y = data[:, 1:]   # shape: [N, T-1, C, H, W]

    # Flatten first two dims to create sample-wise dataset
    X = X.reshape(-1, C, H, W)  # [N*(T-1), C, H, W]
    Y = Y.reshape(-1, C, H, W)  # [N*(T-1), C, H, W]

    return X, Y