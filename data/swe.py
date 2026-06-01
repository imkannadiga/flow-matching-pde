import os
import h5py
import torch
from pathlib import Path

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
    with u(t) as the first channel (state channel) followed by the extra conditions.

    Eval mode (eval=True)
    ---------------------
    Returns one item per trajectory. The evaluator concatenates the current predicted state
    with the extra conditions at each rollout step to match the training conditioning layout.

    Physical time vs flow-matching time
    ------------------------------------
    t_phys = grid/t[t_idx]   — actual simulation clock, optionally in conditioning.
    τ ∈ [0, 1]               — generative transport parameter, handled by FlowMatchingProcessor.

    Data normalization
    ------------------
    When ``normalize_data=True``, samples are z-score normalized using global dataset
    statistics (mean and std computed over all trajectories, timesteps, and spatial points).
    Stats are computed once on the first run and cached to disk alongside the data file
    (``<stem>.global_stats.pt``); subsequent runs with the same file load from cache instantly.
    The cache is invalidated automatically when the data file's modification time changes.

    Both the state channel in conditioning and the target are normalized with the same global
    stats; extra conditions (t_phys_map, coords) are left in their natural range and are NOT
    normalized.
    """

    def __init__(
        self,
        data_path,
        append_physical_time: bool = True,
        normalize_time: bool = True,
        append_coords: bool = False,
        preload: bool = False,
        eval: bool = False,
        normalize_data: bool = True,
        **_,
    ):
        super().__init__(str(data_path), eval=eval)

        self.append_physical_time = append_physical_time
        self.normalize_time = normalize_time
        self.append_coords = append_coords
        self.preload = preload
        self.normalize_data = normalize_data

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
                    traj = torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)
                    self._cache[key] = traj
            else:
                self._cache = None

        if normalize_data:
            self._global_mean, self._global_std = self._compute_or_load_global_stats()
        else:
            self._global_mean = self._global_std = None

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
        self.c_channels = 1 + time_ch + coord_ch
        self.target_channels = 1

    # ------------------------------------------------------------------
    # Global normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stats_cache_path(data_path: str) -> Path:
        p = Path(data_path)
        return p.parent / f"{p.stem}.global_stats.pt"

    def _compute_or_load_global_stats(self) -> tuple[float, float]:
        """Return (mean, std) over the entire dataset, loading from cache when available.

        Uses Welford-style online accumulation so only one trajectory is in memory at a
        time when preload=False.  The cache is keyed on the data file's mtime, so it is
        invalidated automatically if the file is replaced.
        """
        cache_path = self._stats_cache_path(self.data_path)
        data_mtime = os.path.getmtime(self.data_path)

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            if cached.get("source_mtime") == data_mtime:
                return float(cached["mean"]), float(cached["std"])

        print(f"[SWEDataModule] Computing global normalization stats for {self.data_path} ...")

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        if self._cache is not None:
            for traj in self._cache.values():
                vals = traj.double()
                total_sum    += vals.sum().item()
                total_sq_sum += (vals ** 2).sum().item()
                total_count  += vals.numel()
        else:
            with h5py.File(self.data_path, "r") as f:
                for key in self._sample_keys:
                    arr = f[f"{key}/data"][:].astype("float64")
                    total_sum    += arr.sum()
                    total_sq_sum += (arr ** 2).sum()
                    total_count  += arr.size

        mean = total_sum / total_count
        std  = max((total_sq_sum / total_count - mean ** 2) ** 0.5, 1e-8)

        try:
            torch.save({"mean": mean, "std": std, "source_mtime": data_mtime}, cache_path)
        except OSError:
            pass  # read-only filesystem — recompute next time

        return float(mean), float(std)

    @staticmethod
    def _normalize(u: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        return (u - mean) / std

    @staticmethod
    def _denormalize(u: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        return u * std + mean

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

        State channels are z-score normalized using global dataset stats; extra
        conditions (t_phys_map, coords) are left in their natural range and are NOT normalized.
        """
        key, t_idx = self._index[idx]
        traj = self._load_trajectory(key)  # [T, 1, H, W]
        _, _, H, W = traj.shape

        if self.normalize_data:
            u_current = self._normalize(traj[t_idx],     self._global_mean, self._global_std)
            u_next    = self._normalize(traj[t_idx + 1], self._global_mean, self._global_std)
        else:
            u_current = traj[t_idx]
            u_next    = traj[t_idx + 1]

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

        if self.normalize_data:
            norm_traj = self._normalize(traj, self._global_mean, self._global_std)
        else:
            norm_traj = traj

        conditions = torch.stack([
            self._build_extra_conditions(t, H, W)
            for t in range(T - 1)
        ])  # [T-1, C_extra, H, W]

        return {
            "x_0":           norm_traj[0],
            "conditions":    conditions,
            "targets":       norm_traj[1:],
            "time_schedule": self._t_grid[:-1].clone(),
        }
