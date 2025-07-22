from hydra.utils import instantiate
from evaluation.eval_utils import load_model_from_manifest
import torch
from pathlib import Path

class BaseEvaluator:
    def __init__(self, model, data, state_dict_path, device):
        self.model = model
        self.data = data
        self.device = device
        self._load_model(state_dict_path)

    def _load_model(self, state_dict_path):
        save_dir = Path(state_dict_path) 
        self.model = load_model_from_manifest(save_dir, self.model)
        self.model.to(self.device)
        return
    
    def load_data(self):
        raise NotImplementedError("Subclass must implement load_data")
        
    def run(self):
        raise NotImplementedError("Subclasses must implement run")
