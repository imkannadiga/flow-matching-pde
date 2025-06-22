import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import hydra
from omegaconf import DictConfig

from models.FFM import FFMModel
from train import train_model
from general.eval import evaluate_model
from data import load_data

@hydra.main(config_path="../../config/NS-FNO", config_name="config", version_base=None)
def run_task(cfg: DictConfig):
    '''
    FFM with Fourier neural operator (FNO)
    '''
    # Load data using data.py
    print(f"Loading data with configuration: {cfg.data}")
    loader_tr, loader_te = load_data(cfg)
    # Instantiate model from impmorting the relavent model from models
    print(f"Initializing model with configuration: {cfg.model}")
    model = FFMModel(
        modes=cfg.model.modes,
        visch=cfg.model.visch,
        hch=cfg.model.hch,
        pch=cfg.model.pch,
        x_dim=cfg.model.xdim,
        t_scaling=cfg.model.t_scaling,
        kernel_length=cfg.model.lengthscale,
        kernel_variance=cfg.model.var,
        sigma_min=cfg.model.sigma_min,
        device=cfg.model.device
    )
    
    model.to(cfg.model.device)
    
    print("Model initialized successfully.")
    print(model)
    print("started training...")
    train_model(model, cfg, loader_tr, loader_te)
    
    # Evaluate the model and save results in ../results/<task_name>/
    print("Evaluating model...")
    evaluate_model(model, loader_te)
    
if __name__ == "__main__":
    run_task()
    