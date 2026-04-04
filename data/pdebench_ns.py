"""
PDEBench-style Navier-Stokes HDF5 volumes.

Expects a file with a velocity dataset (configurable key) shaped like
``[N_trajectories, T, H, W, C]`` or ``[N, T, C, H, W]``, and scalar attributes
``Re`` / ``nu`` (names configurable) used to build ``batch["params"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch

from data.base import BaseDataModule


def _read_scalar_attr(f: h5py.File, names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name in f.attrs:
            v = f.attrs[name]
            return float(np.asarray(v).reshape(-1)[0])
        for key in f.keys():
            g = f[key]
            if isinstance(g, h5py.Group) and name in g.attrs:
                v = g.attrs[name]
                return float(np.asarray(v).reshape(-1)[0])
    return None


class PDEBenchNSDataModule(BaseDataModule):
    """Loads NS trajectories from a single HDF5 file; exposes ``constant_params`` per file attrs."""

    domain_bounds = ((0.0, 1.0), (0.0, 1.0))

    def __init__(
        self,
        name: str = "pdebench_ns",
        data_dir: str = "",
        filename: Optional[str] = None,
        batch_size: int = 32,
        n_tr: int = 8,
        n_te: int = 2,
        seed: int = 42,
        h5_dataset_key: str = "tensor",
        re_attr_names: Optional[List[str]] = None,
        nu_attr_names: Optional[List[str]] = None,
        **kwargs,
    ):
        re_attr_names = list(re_attr_names or ["Re", "re", "reynolds"])
        nu_attr_names = list(nu_attr_names or ["nu", "viscosity", "Nu"])
        self.h5_dataset_key = h5_dataset_key
        self._re_names = re_attr_names
        self._nu_names = nu_attr_names
        super().__init__(
            name=name,
            data_dir=data_dir,
            filename=filename,
            batch_size=batch_size,
            n_tr=n_tr,
            n_te=n_te,
            seed=seed,
            **kwargs,
        )

    def read_data(self) -> torch.Tensor:
        root = Path(self.cfg.root_dir)
        fn = self.cfg.filename
        if not fn:
            raise ValueError("PDEBenchNSDataModule requires data.filename")
        path = root / fn
        if not path.is_file():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as f:
            Re = _read_scalar_attr(f, self._re_names)
            nu = _read_scalar_attr(f, self._nu_names)
            if self.h5_dataset_key not in f:
                raise KeyError(
                    f"Dataset {self.h5_dataset_key!r} not in {path}; keys={list(f.keys())}"
                )
            arr = np.asarray(f[self.h5_dataset_key][...], dtype=np.float32)

        params_list = [float(Re or 0.0), float(nu or 0.0)]

        # [N, T, H, W, C] or [N, T, C, H, W]
        if arr.ndim == 5:
            if arr.shape[-1] <= 4 and arr.shape[2] > 8:
                # assume [N, T, H, W, C]
                arr = np.transpose(arr, (0, 1, 4, 2, 3))
            # now [N, T, C, H, W]
        else:
            raise ValueError(f"Expected 5D array in HDF5; got shape {arr.shape}")

        self.constant_params = torch.tensor(params_list, dtype=torch.float32)
        return torch.from_numpy(arr)

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 5:
            raise ValueError(f"Expected [N,T,C,H,W]; got {tuple(data.shape)}")
        return data.float()
