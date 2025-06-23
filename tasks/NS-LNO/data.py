import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from omegaconf import DictConfig

def load_data(cfg: DictConfig):
    '''
    Reads data file name from task specific config, reads the data, pre-process it
    Returns the training and testing data loader of batch size
    '''
    # Obtain path from config 
    file_path = os.path.join(cfg.data.root_dir, cfg.data.filename)
    
    dataset = read_data(file_path)
    
    data_processed = preprocess(dataset)
    
    data_tr, data_te = data_processed[:cfg.data.n_tr], data_processed[cfg.data.n_tr:]

    train_loader = DataLoader(data_tr, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(data_te, batch_size=cfg.data.batch_size, shuffle=False)

    return train_loader, test_loader

def load_raw_data(cfg: DictConfig):
    '''
    Loads raw data and reshapes it into a dataset of shape [N, T, C, H, W]
    '''

    # Load file
    file_path = os.path.join(cfg.data.root_dir, cfg.data.filename)
    dataset = read_data(file_path)  # [T, H, W, N]
    
    # Rearrange to [N, T, C, H, W]
    dataset = dataset.permute(3, 0, 1, 2)   # [N, T, H, W]
    dataset = dataset.unsqueeze(2)         # Add channel dim → [N, T, 1, H, W]

    return dataset


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