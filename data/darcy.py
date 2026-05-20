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
        **_kwargs,
    ):
        """
        Args:
            data_path: Path to a single Darcy HDF5 file.
            data_dir: Directory containing multiple Darcy HDF5 files.
            beta: Beta value override.
            append_beta_channel: Append a constant beta spatial map (single-file mode).
            spatial_conditioning: When True, appends beta, x-coord, and y-coord channels.
        """
        effective_path = data_dir if data_dir is not None else data_path
        super().__init__(effective_path)

        self.spatial_conditioning = spatial_conditioning

        if spatial_conditioning:
            if data_dir is not None:
                files = sorted(Path(data_dir).glob("*.hdf5"))
                if not files:
                    raise ValueError(f"No HDF5 files found in {data_dir}")
            else:
                files = [Path(data_path)]
            self._load_with_spatial(files, beta_override=beta)
        else:
            self._load_single(Path(data_path), beta, append_beta_channel)

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
        # c_channels = x_0 (zeros, same size as target) + nu (+ optional beta)
        self.c_channels = self.target_channels + param_ch

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
        # x_0 (zeros) + nu + beta + x-coord + y-coord
        self.c_channels = self.target_channels + 4

        self.beta = None
        self.append_beta_channel = False

    def __len__(self) -> int:
        return self.nu.shape[0]

    def _fetch_data_pair(self, idx: int):
        X_target = self.pressure[idx]
        x_0 = torch.zeros_like(X_target)  # [C_out, D, D] — no physical prior for Darcy

        C = self.nu[idx]  # [1, D, D]

        if self.spatial_conditioning:
            file_idx = self._sample_file_idx[idx]
            beta_map = torch.full_like(C, self._file_betas[file_idx])
            x_map = self._file_x_grids[file_idx]
            y_map = self._file_y_grids[file_idx]
            C = torch.cat([C, beta_map, x_map, y_map], dim=0)  # [4, D, D]

        elif self.beta is not None and self.append_beta_channel:
            beta_map = torch.full_like(C, self.beta)
            C = torch.cat([C, beta_map], dim=0)  # [2, D, D]

        # x_0 is the first channel(s) of the conditioning
        C = torch.cat([x_0, C], dim=0)

        return C, X_target
