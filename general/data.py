import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from omegaconf import DictConfig

def load_data(cfg: DictConfig):
    '''
    generic load data template
    Function reads data file name from task specific config, reads the data, pre-process it (if needed)
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

def read_data(file_path):
    '''
    data specific read logic
    '''
    return NotImplemented

def preprocess(dataset):
    '''
    data pre-processing pipeline (if needed)
    '''
    return dataset