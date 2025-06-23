import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import hydra
from omegaconf import DictConfig

from models.FFM import FFMModel
from models.lno import LFNO
from train import train_model
from eval import evaluate_model
from data import load_data

@hydra.main(config_path="../../config/NS-LNO", config_name="config", version_base=None)
def run_task(cfg: DictConfig):
    '''
    FFM with Local Fourier neural operator (L-FNO)
    '''
    # Load data using data.py
    print(f"Loading data with configuration: {cfg.data}")
    loader_tr, loader_te = load_data(cfg)
    # Instantiate model from impmorting the relavent model from models
    print(f"Initializing model with configuration: {cfg.model}")
    model = LFNO(cfg.model.modes, cfg.model.visch, cfg.model.hch, default_in_shape=cfg.model.default_in_shape,
                 x_dim=cfg.model.xdim, t_scaling=cfg.model.t_scaling)
    model.to(cfg.model.device)
    print(model)
    model_wrapper = FFMModel(model, 
                         kernel_length=cfg.model.lengthscale, 
                         kernel_variance=cfg.model.var, 
                         sigma_min=cfg.model.sigma_min, 
                         device=cfg.model.device)
    print("Model initialized successfully.")
    print(model)
    print("started training...")
    train_model(model_wrapper, cfg, loader_tr, loader_te)
    
    # Evaluate the model and save results in ../results/<task_name>/
    print("Evaluating model...")
    evaluate_model(model_wrapper, cfg, loader_te)
    print("Evaluation completed.")
    
if __name__ == "__main__":
    run_task()
    