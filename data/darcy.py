import hashlib
import os
import re
from pathlib import Path

import h5py
import torch

from data.base import BaseDataModule


class DarcyDataModule(BaseDataModule):
    def __init__(
        self,
        data_path=None,
        data_dir=None,
        beta=None,
        append_beta_channel=True,
        spatial_conditioning=False,
        normalize_data: bool = True,
        **_kwargs,
    ):
        """
        Args:
            data_path: Path to a single Darcy HDF5 file.
            data_dir: Directory containing multiple Darcy HDF5 files.
            beta: Beta value override.
            append_beta_channel: Append a constant beta spatial map (single-file mode).
            spatial_conditioning: When True, appends beta, x-coord, and y-coord channels.
            normalize_data: When True, z-score normalize nu and pressure independently
                using global dataset stats. Stats are computed once on the first run and
                cached to disk (``_darcy_global_stats_<hash>.pt`` in the data directory).
                The cache is invalidated automatically when any source file's mtime changes.
        """
        effective_path = data_dir if data_dir is not None else data_path
        super().__init__(effective_path)

        self.spatial_conditioning = spatial_conditioning
        self.normalize_data = normalize_data

        if spatial_conditioning:
            if data_dir is not None:
                files = sorted(Path(data_dir).glob("*.hdf5"))
                if not files:
                    raise ValueError(f"No HDF5 files found in {data_dir}")
            else:
                files = [Path(data_path)]
            self._load_with_spatial(files, beta_override=beta)
        else:
            files = [Path(data_path)]
            self._load_single(Path(data_path), beta, append_beta_channel)

        if normalize_data:
            stats = self._compute_or_load_stats(files)
            self._nu_mean       = stats["mean_nu"]
            self._nu_std        = stats["std_nu"]
            self._pressure_mean = stats["mean_pressure"]
            self._pressure_std  = stats["std_pressure"]
            self.nu       = (self.nu       - self._nu_mean)       / self._nu_std
            self.pressure = (self.pressure - self._pressure_mean) / self._pressure_std
        else:
            self._nu_mean = self._nu_std = self._pressure_mean = self._pressure_std = None

    @staticmethod
    def _parse_beta_from_filename(filename: str) -> float | None:
        m = re.search(r"beta([\d.]+)", filename)
        return float(m.group(1)) if m else None

    def _load_single(self, path: Path, beta, append_beta_channel: bool) -> None:
        with h5py.File(path, "r") as f:
            self.nu = torch.tensor(f["nu"][:], dtype=torch.float32)
            self.pressure = torch.tensor(f["tensor"][:], dtype=torch.float32)

        if self.nu.dim() == 3:
            self.nu = self.nu.unsqueeze(1)  # [N, 1, D, D]

        self.beta = beta
        self.append_beta_channel = append_beta_channel
        self.target_channels = self.pressure.shape[1]

        param_ch = 2 if (beta is not None and append_beta_channel) else 1
        # c_channels = nu (+ optional beta map)
        self.c_channels = param_ch

        self._file_x_grids = None
        self._file_y_grids = None
        self._file_betas = None
        self._sample_file_idx = None

    def _load_with_spatial(self, files: list[Path], beta_override=None) -> None:
        nu_list, pressure_list = [], []
        self._file_x_grids: list[torch.Tensor] = []
        self._file_y_grids: list[torch.Tensor] = []
        self._file_betas: list[float] = []
        sample_file_indices: list[int] = []

        for file_idx, fpath in enumerate(files):
            beta_val = self._parse_beta_from_filename(fpath.name)
            if beta_val is None:
                if beta_override is None:
                    raise ValueError(
                        f"Cannot determine beta for '{fpath.name}'. "
                        "Either encode it in the filename (e.g. beta1.0) or set 'beta' in the config."
                    )
                beta_val = beta_override

            with h5py.File(fpath, "r") as f:
                nu = torch.tensor(f["nu"][:], dtype=torch.float32)
                pressure = torch.tensor(f["tensor"][:], dtype=torch.float32)
                xc = torch.tensor(f["x-coordinate"][:], dtype=torch.float32)
                yc = torch.tensor(f["y-coordinate"][:], dtype=torch.float32)

            if nu.dim() == 3:
                nu = nu.unsqueeze(1)  # [N, 1, D, D]

            N = nu.shape[0]
            y_grid_2d, x_grid_2d = torch.meshgrid(yc, xc, indexing="ij")
            self._file_x_grids.append(x_grid_2d.unsqueeze(0))
            self._file_y_grids.append(y_grid_2d.unsqueeze(0))
            self._file_betas.append(beta_val)

            nu_list.append(nu)
            pressure_list.append(pressure)
            sample_file_indices.extend([file_idx] * N)

        self.nu = torch.cat(nu_list, dim=0)
        self.pressure = torch.cat(pressure_list, dim=0)
        self._sample_file_idx = sample_file_indices

        self.target_channels = self.pressure.shape[1]
        # nu + beta + x-coord + y-coord
        self.c_channels = 4

        self.beta = None
        self.append_beta_channel = False

    # ------------------------------------------------------------------
    # Global normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stats_cache_path(files: list[Path]) -> Path:
        key = "|".join(str(f.resolve()) for f in sorted(files))
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return files[0].parent / f"_darcy_global_stats_{h}.pt"

    def _compute_or_load_stats(self, files: list[Path]) -> dict:
        """Return global stats for nu and pressure, loading from cache when available.

        The cache is stored alongside the data files and is keyed on their mtimes,
        so it is invalidated automatically if any source file is replaced.
        nu and pressure stats are computed from the already-loaded in-memory tensors.
        """
        cache_path = self._stats_cache_path(files)
        mtimes = [os.path.getmtime(f) for f in files]

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            if cached.get("source_mtimes") == mtimes:
                return cached

        print(f"[DarcyDataModule] Computing global normalization stats ...")

        nu_d   = self.nu.double()
        pres_d = self.pressure.double()

        stats = {
            "mean_nu":       float(nu_d.mean()),
            "std_nu":        float(nu_d.std().clamp(min=1e-8)),
            "mean_pressure": float(pres_d.mean()),
            "std_pressure":  float(pres_d.std().clamp(min=1e-8)),
            "source_mtimes": mtimes,
        }

        try:
            torch.save(stats, cache_path)
        except OSError:
            pass  # read-only filesystem — recompute next time

        return stats

    def __len__(self) -> int:
        return self.nu.shape[0]

    def _fetch_dataset_label(self, idx: int):
        """Return e.g. ``'beta1.0'`` for each sample in multi-file mode; ``None`` in single-file mode."""
        if self._sample_file_idx is None:
            return None
        file_idx = self._sample_file_idx[idx]
        beta_val = self._file_betas[file_idx]
        # Format nicely: strip trailing zeros so 1.0 → "1", 0.5 → "0.5"
        beta_str = f"{beta_val:g}"
        return f"beta{beta_str}"

    def _fetch_data_pair(self, idx: int):
        nu       = self.nu[idx]        # [1, D, D] — already normalized at init if normalize_data
        pressure = self.pressure[idx]

        C = nu

        if self.spatial_conditioning:
            file_idx = self._sample_file_idx[idx]
            beta_map = torch.full_like(nu, self._file_betas[file_idx])
            x_map = self._file_x_grids[file_idx]
            y_map = self._file_y_grids[file_idx]
            C = torch.cat([nu, beta_map, x_map, y_map], dim=0)  # [4, D, D]

        elif self.beta is not None and self.append_beta_channel:
            beta_map = torch.full_like(nu, self.beta)
            C = torch.cat([nu, beta_map], dim=0)  # [2, D, D]

        return C, pressure
