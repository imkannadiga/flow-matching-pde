from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


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

        for opt in ("x_aux", "channel_mins", "channel_maxs"):
            if opt in data_dict and torch.is_tensor(data_dict[opt]):
                data_dict[opt] = data_dict[opt].to(self.device)

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

class FlowMatchingProcessor(DefaultDataProcessor):
    """
    Processor for Flow Matching. 
    Intercepts the data batch, samples time and noise, calculates the 
    intermediate state, and prepares the dictionary for the Trainer.
    """
    def __init__(
        self,
        device,
        append_coords: bool = False,
        domain_bounds = None,
        film_params: bool = False,
        param_keys = None,
        coord_normalize: str = "neg1_1",
        tau_num_points: int = 100,
    ):
        # Initialize the parent class to inherit coordinate/FiLM capabilities
        super().__init__(
            device=device,
            append_coords=append_coords,
            domain_bounds=domain_bounds,
            film_params=film_params,
            param_keys=param_keys,
            coord_normalize=coord_normalize,
        )
        if tau_num_points < 2:
            raise ValueError("tau_num_points must be >= 2")
        self.tau_num_points = int(tau_num_points)
        self._tau_points_cache: Optional[Tuple[torch.device, torch.dtype, torch.Tensor]] = None

    def _get_tau_points(self, dtype: torch.dtype) -> torch.Tensor:
        device = torch.device(self.device)
        if (
            self._tau_points_cache is None
            or self._tau_points_cache[0] != device
            or self._tau_points_cache[1] != dtype
        ):
            tau_points = torch.linspace(
                0.0, 1.0, steps=self.tau_num_points, device=device, dtype=dtype
            )
            self._tau_points_cache = (device, dtype, tau_points)
        return self._tau_points_cache[2]

    def preprocess(self, data_dict, batched=True, step=0):
        """
        Transforms the dataset output into the Flow Matching trajectory.
        """
        # 1. Flexible Key Extraction 
        # (Supports unified {"conditioning", "target"} or legacy {"x", "y"})
        if "target" in data_dict:
            X_target = data_dict.pop("target").to(self.device)
            C = data_dict.pop("conditioning").to(self.device)
        else:
            X_target = data_dict.pop("y").to(self.device)
            C = data_dict.pop("x").to(self.device)

        B = X_target.shape[0]

        # 2. Flow Matching Mathematics
        # Sample base distribution (standard Gaussian noise)
        X_0 = torch.randn_like(X_target)
        
        # Sample flow time tau from a discrete uniform grid in [0, 1]
        tau_points = self._get_tau_points(X_target.dtype)
        tau_idx = torch.randint(0, self.tau_num_points, (B,), device=self.device)
        tau = tau_points[tau_idx]
        tau_spatial = tau
        while tau_spatial.dim() < X_target.dim():
            tau_spatial = tau_spatial.unsqueeze(-1)
            
        # Calculate intermediate state: X_tau = (1 - tau)*X_0 + tau*X_target
        X_tau = (1 - tau_spatial) * X_0 + tau_spatial * X_target
        
        # Calculate target vector field: V = X_target - X_0
        V_target = X_target - X_0

        # 3. Structure for the Trainer & Model
        # Notice we use the key "u" for the primary state tensor. 
        # This triggers the parent class's `apply_model_conditioning` automatically!
        data_dict["x"] = {
            "u": X_tau,              # Noisy state (coords will be appended here if append_coords=True)
            "t": tau,             # 1D time tensor for DiT/UNet time embeddings
            "cond": C                # Physical conditioning (permeability, previous timestep)
        }
        
        # The Trainer will automatically calculate MSE loss against this target
        data_dict["y"] = V_target

        # 4. Parent Logic: Append coordinates or FiLM params if requested
        # data_dict = self.apply_model_conditioning(data_dict)

        return data_dict

    def postprocess(self, output, data_dict, step=0):
        """
        During training, Flow Matching just needs the raw predicted velocity field 
        to compute the MSE against data_dict["y"]. No un-normalizing is strictly 
        required for the loss computation.
        """
        return output, data_dict