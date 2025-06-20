import sys
sys.path.append('../')

import hydra
from omegaconf import DictConfig
from general.data import load_data
from models.SimpleModel import SimpleFlowModel
from general.train import train_model
from general.eval import evaluate_model

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run_task(cfg: DictConfig):
    '''
    Model template to archestrate a task
    '''
    
    # Load data using data.py
    loader_tr, loader_te = load_data(cfg)
    # Instantiate model from impmorting the relavent model from models
    model = SimpleFlowModel(cfg.model.input_dim, cfg.model.hidden_dim)
    
    # Train the model using the training file
    train_metrics = train_model(model, loader_tr, cfg)
    
    print("Training metrics ::: ", train_metrics)
    
    # Evaluate the model and save results in ../results/<task_name>/
    evaluation_res = evaluate_model(model, loader_te)
    print("evaluation metrics :::: ", evaluation_res)