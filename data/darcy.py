"""
2D Darcy flow HDF5 volumes: permeability ``nu`` and pressure ``tensor`` with coordinate vectors.

Filenames like ``2D_Darcy_Flow_beta2.50_Train.hdf5`` optionally supply ``beta``:

- ``batch["params"]`` with shape ``(batch, 1)`` when ``include_beta`` is true (one scalar per sample).
- When ``append_beta_channel`` is true, a constant **beta** map is exposed as ``batch["x_aux"]``
  (not concatenated into ``x``) so flow matching interpolates only ``nu`` vs pressure; the
  trainer stacks ``x_aux`` onto ``u`` after building ``x_τ`` (see ``training/input_conditioning.py``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.base import BaseDataModule, _train_test_sample_indices

_BETA_NAME_RE = re.compile(r"(?i)beta([0-9]+(?:\.[0-9]+)?)")


def parse_darcy_beta_from_filename(filename: str) -> Optional[float]:
    """Parse ``beta`` from a name like ``2D_Darcy_Flow_beta2.50_Train.hdf5``."""
    m = _BETA_NAME_RE.search(Path(filename).name)
    if not m:
        return None
    return float(m.group(1))


class DarcyPairDataset(Dataset):
    """``x`` = nu field; optional ``x_aux`` (e.g. beta plane); optional ``params``."""

    def __init__(
        self,
        x: torch.Tensor,
        tensor: torch.Tensor,
        per_sample_params: Optional[torch.Tensor] = None,
        x_aux: Optional[torch.Tensor] = None,
    ):
        if x.shape[0] != tensor.shape[0]:
            raise ValueError(f"x N={x.shape[0]} != tensor N={tensor.shape[0]}")
        self.x = x
        self.tensor = tensor
        self.per_sample_params = per_sample_params
        self.x_aux = x_aux
        if per_sample_params is not None and per_sample_params.shape[0] != x.shape[0]:
            raise ValueError(
                f"per_sample_params N={per_sample_params.shape[0]} != x N={x.shape[0]}"
            )
        if x_aux is not None and x_aux.shape[0] != x.shape[0]:
            raise ValueError(f"x_aux N={x_aux.shape[0]} != x N={x.shape[0]}")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {"x": self.x[idx], "y": self.tensor[idx]}
        if self.x_aux is not None:
            out["x_aux"] = self.x_aux[idx]
        if self.per_sample_params is not None:
            out["params"] = self.per_sample_params[idx].clone()
        return out


class DarcyFlowDataModule(BaseDataModule):
    """
    Loads ``nu`` (N, D, D) and ``tensor`` (N, C, D, D) from one or many HDF5 files.

    Train/test split is over **samples** (not time): ``n_tr`` / ``n_te`` count samples.

    ``domain_bounds`` are ``((x_min, x_max), (y_min, y_max))`` from the coordinate datasets.

    With ``load_all_hdf5``, every ``*.hdf5`` and ``*.h5`` under ``data_dir`` is loaded (sorted
    by name), samples are concatenated, and per-file ``beta`` (from the filename) is attached
    to each row (``params`` and/or an extra ``x`` channel).
    """

    def __init__(
        self,
        name: str = "darcy",
        data_dir: str = "",
        filename: Optional[str] = None,
        batch_size: int = 32,
        n_tr: int = 8000,
        n_te: int = 2000,
        seed: int = 42,
        key_nu: str = "nu",
        key_tensor: str = "tensor",
        key_x_coord: str = "x-coordinate",
        key_y_coord: str = "y-coordinate",
        include_beta: bool = True,
        load_all_hdf5: bool = False,
        append_beta_channel: bool = True,
        require_beta_in_filename: bool = False,
        **kwargs,
    ):
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
        self.key_nu = key_nu
        self.key_tensor = key_tensor
        self.key_x_coord = key_x_coord
        self.key_y_coord = key_y_coord
        self.include_beta = include_beta
        self.load_all_hdf5 = load_all_hdf5
        self.append_beta_channel = append_beta_channel
        self.require_beta_in_filename = require_beta_in_filename
        self.domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.0, 1.0),
            (0.0, 1.0),
        )

    def _list_h5_paths(self) -> List[Path]:
        root = Path(self.cfg.root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"data_dir is not a directory: {root}")
        if self.load_all_hdf5:
            paths = sorted({*root.glob("*.hdf5"), *root.glob("*.h5")})
            paths = [p for p in paths if p.is_file()]
            if not paths:
                raise ValueError(f"No *.hdf5 or *.h5 files under {root}")
            return paths
        fn = self.cfg.filename
        if not fn:
            raise ValueError(
                "Set data.filename, or set data.load_all_hdf5=true to glob the folder."
            )
        path = root / fn
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]

    def _load_fields_from_path(
        self, path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = (self.key_nu, self.key_tensor, self.key_x_coord, self.key_y_coord)
        arrays = self.load_hdf5_arrays(path, keys)
        nu = arrays[self.key_nu].float()
        tensor = arrays[self.key_tensor].float()
        xc = arrays[self.key_x_coord].float().reshape(-1)
        yc = arrays[self.key_y_coord].float().reshape(-1)
        if nu.dim() != 3:
            raise ValueError(f"Expected nu (N, D, D); got {tuple(nu.shape)}")
        if tensor.dim() != 4:
            raise ValueError(f"Expected tensor (N, C, D, D); got {tuple(tensor.shape)}")
        d = nu.shape[-1]
        if nu.shape[1] != d or tensor.shape[-1] != d or tensor.shape[-2] != d:
            raise ValueError(
                f"Spatial dims mismatch: nu {tuple(nu.shape)}, tensor {tuple(tensor.shape)}"
            )
        if xc.numel() != d or yc.numel() != d:
            raise ValueError(
                f"Coordinate length must match D={d}; got x {xc.numel()}, y {yc.numel()}"
            )
        nu = nu.unsqueeze(1)
        return nu, tensor, xc, yc

    def _load_merged(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Returns ``x`` (N, 1, D, D) permeability, ``tensor``, ``xc``, ``yc``,
        optional ``per_sample_params`` (N, 1), and optional ``x_aux`` (N, 1, D, D) beta maps.
        """
        paths = self._list_h5_paths()
        multi = len(paths) > 1
        parts_x: List[torch.Tensor] = []
        parts_x_aux: List[torch.Tensor] = []
        parts_y: List[torch.Tensor] = []
        parts_beta: List[torch.Tensor] = []
        xc0: Optional[torch.Tensor] = None
        yc0: Optional[torch.Tensor] = None
        d_grid: Optional[int] = None

        for path in paths:
            nu1, tensor1, xc, yc = self._load_fields_from_path(path)
            beta: Optional[float] = None
            if self.include_beta:
                beta = parse_darcy_beta_from_filename(path.name)

            need_beta = (multi and self.include_beta) or self.require_beta_in_filename
            if beta is None and need_beta:
                raise ValueError(
                    f"Could not parse beta from filename {path.name!r}; "
                    f"expected a substring like 'beta1.5'. "
                    f"Multi-file loads with include_beta=true require beta in every filename."
                )

            if xc0 is None:
                xc0, yc0 = xc, yc
                d_grid = nu1.shape[-1]
            else:
                assert yc0 is not None and d_grid is not None
                if not torch.allclose(xc, xc0) or not torch.allclose(yc, yc0):
                    raise ValueError(
                        f"x/y coordinates differ from first file; offending file: {path}"
                    )
                if nu1.shape[-1] != d_grid:
                    raise ValueError(
                        f"Spatial resolution D differs across files ({d_grid} vs {nu1.shape[-1]}): {path}"
                    )

            parts_x.append(nu1)
            if self.include_beta and self.append_beta_channel and beta is not None:
                nloc, _, d, _ = nu1.shape
                bch = torch.full(
                    (nloc, 1, d, d),
                    beta,
                    dtype=nu1.dtype,
                    device=nu1.device,
                )
                parts_x_aux.append(bch)

            parts_y.append(tensor1)
            if self.include_beta and beta is not None:
                nloc = nu1.shape[0]
                parts_beta.append(torch.full((nloc,), beta, dtype=torch.float32))

        x_cat = torch.cat(parts_x, dim=0)
        y_cat = torch.cat(parts_y, dim=0)
        assert xc0 is not None and yc0 is not None
        per_sp: Optional[torch.Tensor] = None
        if self.include_beta and parts_beta:
            per_sp = torch.cat(parts_beta, dim=0).unsqueeze(1)
        x_aux_cat: Optional[torch.Tensor] = (
            torch.cat(parts_x_aux, dim=0) if parts_x_aux else None
        )
        return x_cat, y_cat, xc0, yc0, per_sp, x_aux_cat

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Unused for Darcy; ``get_dataloaders`` builds loaders directly."""
        return data

    def _build_split_loaders(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        per_sp: Optional[torch.Tensor],
        x_aux: Optional[torch.Tensor],
    ):
        self.constant_params = None
        n = x.shape[0]
        train_idx, test_idx = _train_test_sample_indices(
            n, int(self.cfg.n_tr), int(self.cfg.n_te), self.seed
        )
        train_idx_t = torch.tensor(train_idx, dtype=torch.long)
        test_idx_t = torch.tensor(test_idx, dtype=torch.long)
        if per_sp is not None:
            tr_p, te_p = per_sp[train_idx_t], per_sp[test_idx_t]
        else:
            tr_p, te_p = None, None
        if x_aux is not None:
            tr_a, te_a = x_aux[train_idx_t], x_aux[test_idx_t]
        else:
            tr_a, te_a = None, None
        data_tr = DarcyPairDataset(x[train_idx_t], y[train_idx_t], tr_p, tr_a)
        data_te = DarcyPairDataset(x[test_idx_t], y[test_idx_t], te_p, te_a)
        train_loader = DataLoader(data_tr, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(data_te, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_dataloaders(self):
        x, tensor, xc, yc, per_sp, x_aux = self._load_merged()
        self.domain_bounds = (
            (float(xc.min().item()), float(xc.max().item())),
            (float(yc.min().item()), float(yc.max().item())),
        )
        return self._build_split_loaders(x, tensor, per_sp, x_aux)

    def get_testing_data(self, n_samples: Optional[int] = None, data_loader: bool = True):
        x, tensor, xc, yc, per_sp, x_aux = self._load_merged()
        self.domain_bounds = (
            (float(xc.min().item()), float(xc.max().item())),
            (float(yc.min().item()), float(yc.max().item())),
        )
        self.constant_params = None

        n = x.shape[0]
        _, test_idx = _train_test_sample_indices(
            n, int(self.cfg.n_tr), int(self.cfg.n_te), self.seed
        )
        test_idx_t = torch.tensor(test_idx, dtype=torch.long)
        if per_sp is not None:
            te_p = per_sp[test_idx_t]
        else:
            te_p = None
        if x_aux is not None:
            te_a = x_aux[test_idx_t]
        else:
            te_a = None
        test_dataset = DarcyPairDataset(x[test_idx_t], tensor[test_idx_t], te_p, te_a)

        if n_samples is not None and n_samples < len(test_dataset):
            g = torch.Generator()
            g.manual_seed(self.seed + 1)
            indices = torch.randperm(len(test_dataset), generator=g)[:n_samples].tolist()
            test_dataset = torch.utils.data.Subset(test_dataset, indices)

        if data_loader:
            return test_dataset, DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        return test_dataset
