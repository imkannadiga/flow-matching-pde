from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from data.coords import make_coord_grid


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
        device,
        append_coords: bool = False,
        domain_bounds: Optional[Sequence[Sequence[float]]] = None,
        film_params: bool = False,
        param_keys: Optional[Sequence[str]] = None,
        coord_normalize: str = "neg1_1",
    ):
        """
        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        append_coords : bool
            If True, concatenate ``make_coord_grid`` channels to ``u`` in ``sample["x"]``.
        domain_bounds : sequence of (min, max) per spatial axis
        film_params : bool
            If True, pass ``sample["params"]`` through to ``sample["x"]["params"]`` for FiLM.
        param_keys : optional names to build ``params`` vectors from ``sample["metadata"]`` dicts
        coord_normalize : passed to ``make_coord_grid``
        """
        super().__init__()
        self.device = device
        self.model = None
        self.append_coords = bool(append_coords)
        self.film_params = bool(film_params)
        self.param_keys: List[str] = list(param_keys or [])
        self.coord_normalize = coord_normalize
        if domain_bounds is not None:
            self.domain_bounds: List[Tuple[float, float]] = [
                (float(a), float(b)) for a, b in domain_bounds
            ]
        else:
            self.domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
        self._coord_cache: Optional[Tuple[Tuple[int, int], torch.Tensor]] = None

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

        if "params" in data_dict and torch.is_tensor(data_dict["params"]):
            data_dict["params"] = data_dict["params"].to(self.device)

        if (
            self.param_keys
            and "metadata" in data_dict
            and "params" not in data_dict
        ):
            meta = data_dict["metadata"]
            if torch.is_tensor(meta):
                raise ValueError("metadata must be a dict or list of dicts for param_keys")
            if isinstance(meta, dict):
                vec = torch.tensor(
                    [float(meta[k]) for k in self.param_keys],
                    device=self.device,
                    dtype=x.dtype,
                )
                if x.dim() == 4:
                    vec = vec.unsqueeze(0).expand(x.shape[0], -1)
                data_dict["params"] = vec
            elif isinstance(meta, list):
                rows = []
                for m in meta:
                    rows.append([float(m[k]) for k in self.param_keys])
                data_dict["params"] = torch.tensor(
                    rows, device=self.device, dtype=x.dtype
                )

        return data_dict

    def _batched_coord_grid(
        self, batch_size: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (h, w)
        if self._coord_cache is None or self._coord_cache[0] != key:
            g = make_coord_grid(
                self.domain_bounds,
                (h, w),
                normalize=self.coord_normalize,
                device=device,
                dtype=dtype,
            )
            self._coord_cache = (key, g)
        g = self._coord_cache[1]
        return g.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def apply_model_conditioning(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """After a paradigm builds ``sample["x"]`` with a ``u`` tensor, concat coords / params."""
        if not isinstance(sample.get("x"), dict) or "u" not in sample["x"]:
            return sample
        u = sample["x"]["u"]
        if u.dim() != 4:
            return sample
        b, _, h, w = u.shape
        if self.append_coords:
            coords = self._batched_coord_grid(b, h, w, u.device, u.dtype)
            sample["x"]["u"] = torch.cat([u, coords], dim=1)
        if self.film_params and "params" in sample:
            p = sample["params"]
            if not torch.is_tensor(p):
                raise TypeError("film_params requires sample['params'] to be a tensor")
            sample["x"]["params"] = p
        return sample

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