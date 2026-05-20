from pathlib import Path

import h5py
import torch

from data.base import BaseDataModule


class SWEDataModule(BaseDataModule):
    """Dataset for the Shallow Water Equations (SWE).

    HDF5 layout — each sample lives under a zero-padded key (e.g. "0999"):
        XXXX/data        (T, H, W, 1) float32   water-height trajectory
        XXXX/grid/t      (T,)         float32   physical time coordinates
        XXXX/grid/x      (H,)         float32   x spatial coordinates
        XXXX/grid/y      (W,)         float32   y spatial coordinates

    Training mode (eval=False)
    --------------------------
    Produces N × (T-1) training pairs, one per consecutive-step transition u(t) → u(t+Δt).
    Each conditioning tensor C = [u(t), t_phys_map, (x_coord, y_coord)] is returned as "x",
    with u(t) as the first channel so the processor can apply optional corruption.

    Eval mode (eval=True)
    ---------------------
    Returns one item per trajectory. The evaluator concatenates the current predicted state
    with the extra conditions at each rollout step to match the training conditioning layout.

    Physical time vs flow-matching time
    ------------------------------------
    t_phys = grid/t[t_idx]   — actual simulation clock, optionally in conditioning.
    τ ∈ [0, 1]               — generative transport parameter, handled by FlowMatchingProcessor.
    """

    def __init__(
        self,
        data_path,
        append_physical_time: bool = True,
        normalize_time: bool = True,
        append_coords: bool = False,
        preload: bool = False,
        eval: bool = False,
        **_kwargs,
    ):
        super().__init__(str(data_path), eval=eval)

        self.append_physical_time = append_physical_time
        self.normalize_time = normalize_time
        self.append_coords = append_coords
        self.preload = preload

        with h5py.File(self.data_path, "r") as f:
            self._sample_keys: list[str] = sorted(k for k in f.keys())
            if not self._sample_keys:
                raise ValueError(f"No top-level groups found in {data_path}")

            first = self._sample_keys[0]
            t_raw = torch.tensor(f[f"{first}/grid/t"][:], dtype=torch.float32)
            x_grid = torch.tensor(f[f"{first}/grid/x"][:], dtype=torch.float32)
            y_grid = torch.tensor(f[f"{first}/grid/y"][:], dtype=torch.float32)

            T = len(t_raw)

            self._index: list[tuple[str, int]] = [
                (key, t) for key in self._sample_keys for t in range(T - 1)
            ]

            if preload:
                self._cache: dict[str, torch.Tensor] = {}
                for key in self._sample_keys:
                    arr = f[f"{key}/data"][:]
                    self._cache[key] = (
                        torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)
                    )
            else:
                self._cache = None

        if normalize_time:
            t_min, t_max = t_raw[0], t_raw[-1]
            span = (t_max - t_min).clamp(min=1e-8)
            self._t_grid = (t_raw - t_min) / span
        else:
            self._t_grid = t_raw

        y2d, x2d = torch.meshgrid(y_grid, x_grid, indexing="ij")
        self._x_coord = x2d.unsqueeze(0)  # [1, H, W]
        self._y_coord = y2d.unsqueeze(0)  # [1, H, W]

        time_ch = 1 if append_physical_time else 0
        coord_ch = 2 if append_coords else 0
        # c_channels = u_current (state) + extra conditions
        self.c_channels = 1 + time_ch + coord_ch
        self.target_channels = 1

    def _load_trajectory(self, key: str) -> torch.Tensor:
        if self._cache is not None:
            return self._cache[key]
        with h5py.File(self.data_path, "r") as f:
            arr = f[f"{key}/data"][:]
        return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, 1, H, W)

    def _build_extra_conditions(self, t_idx: int, H: int, W: int) -> torch.Tensor:
        """Build non-state conditioning channels for one physical time step.

        Returns [C_extra, H, W] containing optional t_phys broadcast and coord maps.
        This is the single source of truth for extra-channel order so training and
        eval rollout always produce identical layouts.
        """
        parts = []
        if self.append_physical_time:
            t_val = self._t_grid[t_idx].item()
            parts.append(torch.full((1, H, W), t_val, dtype=torch.float32))
        if self.append_coords:
            parts.append(self._x_coord[:, :H, :W])
            parts.append(self._y_coord[:, :H, :W])
        if not parts:
            return torch.zeros(0, H, W, dtype=torch.float32)
        return torch.cat(parts, dim=0)

    def __len__(self) -> int:
        if self.eval:
            return self._trajectory_count()
        return len(self._index)

    def _fetch_data_pair(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (C, X_target) for a single consecutive-step transition.

        C  [c_channels, H, W]  — conditioning: [u(t_phys), t_phys_map?, x_coord?, y_coord?]
        X_target  [1, H, W]    — ground-truth next state u(t_phys + Δt)
        """
        key, t_idx = self._index[idx]
        traj = self._load_trajectory(key)  # [T, 1, H, W]
        _, _, H, W = traj.shape

        u_current = traj[t_idx]       # [1, H, W] — physical prior state (goes into cond as x_0)
        u_next = traj[t_idx + 1]      # [1, H, W] — target
        extra = self._build_extra_conditions(t_idx, H, W)  # [C_extra, H, W]

        C = torch.cat([u_current, extra], dim=0) if extra.shape[0] > 0 else u_current

        return C, u_next

    def _trajectory_count(self) -> int:
        return len(self._sample_keys)

    def _fetch_trajectory(self, idx: int) -> dict:
        """Return a full trajectory for auto-regressive rollout evaluation.

        Returns
        -------
        dict with keys:
          'x_0'           [1, H, W]             true initial state u(t=0)
          'conditions'    [T-1, C_extra, H, W]  extra conditioning per step (no state channel)
          'targets'       [T-1, 1, H, W]        ground-truth states u(t=1)..u(t=T-1)
          'time_schedule' [T-1]                 physical time value at each step

        The evaluator prepends the current predicted state to conditions[t] at each
        rollout step, matching the layout produced by _fetch_data_pair during training.
        """
        key = self._sample_keys[idx]
        traj = self._load_trajectory(key)  # [T, 1, H, W]
        T, _, H, W = traj.shape

        conditions = torch.stack([
            self._build_extra_conditions(t, H, W)
            for t in range(T - 1)
        ])  # [T-1, C_extra, H, W]

        return {
            "x_0":           traj[0],
            "conditions":    conditions,
            "targets":       traj[1:],
            "time_schedule": self._t_grid[:-1].clone(),
        }
