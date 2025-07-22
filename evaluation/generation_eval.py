import torch
from evaluation.base import BaseEvaluator
from .eval_utils import get_pred_seq_with_comparision
from pathlib import Path

class GenerationEvaluator(BaseEvaluator):
    def __init__(self, model, data, state_dict_path, device, n_steps, n_samples, save_path, *kwargs):
        super().__init__(model, data, state_dict_path, device)
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
    
    def load_data(self):
        test_data = self.data.get_testing_data(self.n_samples, data_loader=False)
        return test_data

    
    def run(self):
        ## 1. Get 'n' initial conditions from the dataset
        testdata = self.load_data()
        ## 2. Generate n_samples random initial conditions with shape testdata.shape[2:]
        spatial_shape = testdata.shape[2:]  # e.g., (64, 64)
        shape = (self.n_samples,) + spatial_shape
        random_initial_conditions = torch.randn(shape)
        
        ## 3. Get outputs for all of the random conditions
        # TODO : Fix the implementation for KDE and spectrum calculations
        
        return