from abc import ABCMeta, abstractmethod

import torch
from neuralop.training.patching import MultigridPatching2D
import numpy as np

import torch
from neuralop.training.patching import MultigridPatching2D

from util.gaussian_process import GPPrior
from util.util import make_grid, reshape_for_batchwise


class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with PyTorch convention
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - forward(self, x):
            forward pass providing that a model has been wrapped
        """
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    # default wrap method
    def wrap(self, model):
        self.model = model
        return self
    
    # default train and eval methods
    def train(self, val: bool=True):
        super().train(val)
        if self.model is not None:
            self.model.train()
    
    def eval(self):
        super().eval()
        if self.model is not None:
            self.model.eval()

    @abstractmethod
    def forward(self, x):
        pass

class DefaultDataProcessor(DataProcessor):
    """DefaultDataProcessor is a simple processor 
    to pre/post process data before training/inferencing a model.
    """
    def __init__(
        self,
        device
    ):
        """
        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        """
        super().__init__()
        self.device=device
        self.model = None

    def to(self, device):
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        """preprocess a batch of data into the format
        expected in model's forward call

        By default, training loss is computed on normalized out and y
        and eval loss is computed on unnormalized out and y

        Parameters
        ----------
        data_dict : dict
            input data dictionary with at least
            keys 'x' (inputs) and 'y' (ground truth)
        batched : bool, optional
            whether data contains a batch dim, by default True

        Returns
        -------
        dict
            preprocessed data_dict
        """
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)
        
        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict):
        """postprocess model outputs and data_dict
        into format expected by training or val loss

        By default, training loss is computed on normalized out and y
        and eval loss is computed on unnormalized out and y

        Parameters
        ----------
        output : torch.Tensor
            raw model outputs
        data_dict : dict
            dictionary containing single batch
            of data

        Returns
        -------
        out, data_dict
            postprocessed outputs and data dict
        """
        
        return output, data_dict
    
    def forward(self, x):
        return self.preprocess(x)

class FlowMatchingDataProcessor(DefaultDataProcessor):
    """
    A data processor for flow matching tasks.
    
    This class provides the functionality to convert raw data -> {"x":x, "y":y, "t":t}
    into a format suitable for flow matching tasks -> {"x":(x,t), "y":y}.
    """
    
    def __init__(self, kernel_length, kernel_variance, vp, sigma_min, device):
        
        super().__init__(device)
        
        self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device=device)
        self.device = device
        self.vp = vp
        self.sigma_min = sigma_min
        if self.vp:
            self.alpha, self.dalpha = self.construct_alpha()

    def construct_alpha(self):
        def alpha(t):
            return torch.cos((t + 0.08)/2.16 * np.pi).to(self.device)
        def dalpha(t):
            return -np.pi/2.16 * torch.sin((t + 0.08)/2.16 * np.pi).to(self.device)
        return alpha, dalpha
    
    def simulate(self, t, x_data):
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, *dims]
        # samples from p_t(x | x_data)

        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        # Sample from prior GP
        query_points = make_grid(dims)
        
        noise = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=n_channels)

        # Construct mean/variance parameters
        t = reshape_for_batchwise(t, 1 + n_dims)
        
        if self.vp:
            mu = self.alpha(1-t) * x_data
            sigma = torch.sqrt((1 - self.alpha(1-t)**2))
        else:
            mu = t * x_data
            sigma = 1. - (1. - self.sigma_min) * t

        samples = mu + sigma * noise

        assert samples.shape == x_data.shape
        return samples
    
    def get_conditional_fields(self, t, x_data, x_noisy):
        # computes v_t(x_noisy | x_data)
        # x_data, x_noisy: (batch_size, n_channels, *dims)

        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)

        t = reshape_for_batchwise(t, 1 + n_dims)
        if self.vp:
            conditional_fields = (self.dalpha(1-t)/(1 - self.alpha(1-t)**2)) * (self.alpha(1-t)*x_noisy - x_data)
        else:
            c = 1. - (1. - self.sigma_min) * t
            conditional_fields = ( x_data - (1. - self.sigma_min) * x_noisy ) / c

        return conditional_fields

    def preprocess(self, sample):
        """
        Preprocess the sample for flow matching tasks.
        
        Samples a random time from [0,1) 
        
        Args:
            sample (dict): The input sample containing data to be processed.
        
        Returns:
            dict: The preprocessed sample.
        """
        
        sample = super().preprocess(sample)
        
        batch_size = sample["x"].shape[0]
        
        # Sample a random time t from [0, 1)
        t = torch.rand(batch_size, device=self.device)
        
        x_noisy = self.simulate(t=t, x_data=sample["x"])
        
        # Get conditional vector fields
        target = self.get_conditional_fields(t, sample["x"], x_noisy)

        processed_sample = {
            "x":{
                "u": x_noisy,
                "t":t
            },
            "y": target
        }
        
        return processed_sample