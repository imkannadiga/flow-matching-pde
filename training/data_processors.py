from abc import ABCMeta, abstractmethod

import torch


class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    """Base class for data processors used before and after model forward passes."""

    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    def wrap(self, model):
        self.model = model
        return self

    def train(self, val: bool = True):
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


class FlowMatchingProcessor(DataProcessor):
    """Preprocesses batches for flow matching training.

    Conditioning C from the dataset has x_0 as its first `state_channels` channels,
    followed by any physical-time maps or spatial coords. The processor:
      1. Draws Gaussian noise x_noise with the same shape as x_1 (target).
      2. Interpolates: x_tau = (1 - tau) * x_noise + tau * x_1.
      3. Computes velocity target: v_target = x_1 - x_noise.

    Model receives: u=x_tau, cond=C, t=tau.
    """

    def __init__(
        self,
        device,
        state_channels: int = 1,
        **_kwargs,
    ):
        super().__init__()
        self.device = device
        self.model = None
        self.state_channels = int(state_channels)

    def to(self, device):
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True, step=0):
        x_1 = data_dict.pop("y").to(self.device)   # [B, C_out, H, W]
        C = data_dict.pop("x").to(self.device)      # [B, C_cond, H, W], first state_channels = x_0
        
        B = x_1.shape[0]
        x_noise = torch.randn_like(x_1)
        tau = torch.rand(B, device=self.device, dtype=x_1.dtype)
        tau_s = tau.view(B, 1, 1, 1)

        x_tau = (1 - tau_s) * x_noise + tau_s * x_1
        v_target = x_1 - x_noise

        data_dict["x"] = {"u": x_tau, "t": tau, "cond": C}
        data_dict["y"] = v_target
        return data_dict

    def postprocess(self, output, data_dict, step=0):
        return output, data_dict

    def forward(self, x):
        return self.preprocess(x)
