from evaluation.base import BaseEvaluator
from .eval_utils import get_pred_seq_with_comparision
from pathlib import Path

class SimulationEvaluator(BaseEvaluator):
    def __init__(self, model, data, state_dict_path, device, n_steps, n_samples, save_path, *kwargs):
        super().__init__(model, data, state_dict_path, device)
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
    
    def load_data(self):
        test_loader = self.data.get_testing_data(self.n_samples)
        return test_loader
    
    def run(self):
        ## 1. Get 'n' initial conditions from the dataset
        testdata_loader = self.load_data()
        ## 2. Get simulation of 'n_steps' for all the ICs using PDEInt
        get_pred_seq_with_comparision(test_loader=testdata_loader, 
                                      n_steps=self.n_steps, 
                                      model=self.model, 
                                      device=self.device,
                                      save_path=self.save_path)
        
        return