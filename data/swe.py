from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from data.base import BaseDataModule


class SWEDataModule(BaseDataModule):
    """Dataset for the Shallow Water Equations (SWE).

    HDF5 layout — each sample lives under a zero-padded key (e.g. "0999"):
        XXXX/data        (T, H, W, 1) float32   water-height trajectory
        XXXX/grid/t      (T,)         float32   physical time coordinates
        XXXX/grid/x      (H,)         float32   x spatial coordinates
        XXXX/grid/y      (W,)         float32   y spatial coordinates

    Training pairs
    --------------
    Each sample is a trajectory of T=101 snapshots.  We produce N × (T-1)
    independent training pairs, one per consecutive-step transition u(t) → u(t+Δt).
    The flat index maps a scalar idx to a (sample_key, t_idx) tuple.

    Physical time vs flow-matching time
    ------------------------------------
    Two distinct notions of "time" coexist in this repo and must never be confused:

    1. Physical PDE time  t_phys = grid/t[t_idx]
       The actual simulation clock (e.g. 0.0–10.0 s).  Optionally appended as a
       broadcast spatial channel inside the conditioning tensor C so the model
       knows *where in the physical trajectory* it is operating.
       Lives in C[1] (if append_physical_time=True).

    2. Flow-matching time  τ ∈ [0, 1]
       The generative transport parameter sampled by FlowMatchingProcessor.
       Parameterises the straight-line path from X_0 → X_target during training
       and the ODE integration during inference.
       Passed to the model as sample["x"]["t"] by the processor.
       This dataset class never touches τ.

    _make_x0 — the key difference from Darcy
    -----------------------------------------
    Darcy is a static PDE: X_0 = zeros (flow generates a solution from scratch).

    For SWE, X_0 = u(t_phys) — the previous physical state.  The flow-matching
    path therefore goes from the real prior state to the next physical state, which
    is physically meaningful and keeps the generative task small (one time-step
    perturbation, not a generation from noise).
    """

    def __init__(
        self,
        data_path,
        append_physical_time: bool = True,
        normalize_time: bool = True,
        append_coords: bool = False,
        preload: bool = False,
        **_kwargs,
    ):
        """
        Parameters
        ----------
        data_path : str | Path
            Path to the HDF5 file.
        append_physical_time : bool
            Append a spatial map of t_phys as an extra conditioning channel.
            Lets the model distinguish early vs. late trajectory steps.
        normalize_time : bool
            If True, normalise grid/t to [0, 1] before broadcasting.
            Recommended — keeps the time channel on the same scale as the state.
        append_coords : bool
            Append x- and y-coordinate maps as two additional channels (like Darcy
            spatial conditioning).  Useful for non-uniform grids.
        preload : bool
            Load all trajectories into RAM at init time.  Fast per-item access but
            requires enough memory (N × T × H × W × 4 bytes for float32).
            If False, opens the HDF5 file on each __getitem__ call (safe for
            multiprocessing DataLoaders; the OS page cache makes repeat reads cheap).
        """
        super().__init__(str(data_path))

        self.append_physical_time = append_physical_time
        self.normalize_time = normalize_time
        self.append_coords = append_coords
        self.preload = preload

        # ------------------------------------------------------------------
        # Scan the file: collect keys, read shared grid, build flat index
        # ------------------------------------------------------------------
        with h5py.File(self.data_path, "r") as f:
            sample_keys: list[str] = sorted(k for k in f.keys())
            if not sample_keys:
                raise ValueError(f"No top-level groups found in {data_path}")

            # Grid is shared across all samples; read it once from the first sample
            first = sample_keys[0]
            t_raw = torch.tensor(f[f"{first}/grid/t"][:], dtype=torch.float32)  # [T]
            x_grid = torch.tensor(f[f"{first}/grid/x"][:], dtype=torch.float32)  # [H]
            y_grid = torch.tensor(f[f"{first}/grid/y"][:], dtype=torch.float32)  # [W]

            T = len(t_raw)

            # Flat (sample_key, t_idx) index — one entry per consecutive pair
            # Total length = N_samples × (T - 1)
            self._index: list[tuple[str, int]] = [
                (key, t) for key in sample_keys for t in range(T - 1)
            ]

            if preload:
                self._cache: dict[str, torch.Tensor] = {}
                for key in sample_keys:
                    # HDF5 layout is (T, H, W, 1) — channel-last
                    arr = f[f"{key}/data"][:]
                    # Convert to PyTorch convention: (T, 1, H, W)
                    self._cache[key] = (
                        torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)
                    )
            else:
                self._cache = None

        # Physical time grid — optionally normalise to [0, 1]
        if normalize_time:
            t_min, t_max = t_raw[0], t_raw[-1]
            span = (t_max - t_min).clamp(min=1e-8)
            self._t_grid = (t_raw - t_min) / span   # [T], values ∈ [0, 1]
        else:
            self._t_grid = t_raw                     # [T], raw physical values

        # Precompute spatial coord grids once; reused by every _fetch_data_pair call
        # Shape [1, H, W] — ready to cat along the channel dim
        y2d, x2d = torch.meshgrid(y_grid, x_grid, indexing="ij")   # [H, W]
        self._x_coord = x2d.unsqueeze(0)   # [1, H, W]
        self._y_coord = y2d.unsqueeze(0)   # [1, H, W]

        # Channel counts — consumed by train.py's _infer_model_channels to
        # auto-configure in_channels and out_channels on the model
        state_ch = 1
        time_ch  = 1 if append_physical_time else 0
        coord_ch = 2 if append_coords else 0
        self.c_channels     = state_ch + time_ch + coord_ch
        self.target_channels = state_ch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trajectory(self, key: str) -> torch.Tensor:
        """Return the full trajectory tensor [T, 1, H, W] for one sample.

        Preloaded mode: O(1) dict lookup + slice — no I/O.
        Lazy mode: opens the HDF5 file per call.  Safe for multiprocessing
        because each DataLoader worker is a separate process with its own
        file descriptor.  Note that __getitem__ calls this twice per sample
        (once in _fetch_data_pair, once in _make_x0); the OS page cache makes
        the second read negligible.  If this matters, enable preload=True.
        """
        if self._cache is not None:
            return self._cache[key]
        with h5py.File(self.data_path, "r") as f:
            arr = f[f"{key}/data"][:]                            # (T, H, W, 1)
        return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, 1, H, W)

    # ------------------------------------------------------------------
    # BaseDataModule interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def _fetch_data_pair(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (C, X_target) for a single consecutive-step transition.

        C  [c_channels, H, W]  — conditioning tensor:
            ch 0       : u(t_phys)      current physical state
            ch 1       : t_phys map     PHYSICAL PDE time (not flow-matching τ!)
                         broadcast as a constant spatial map so the model knows
                         its position in the physical trajectory
            ch 2, 3    : x, y coords    (only if append_coords=True)

        X_target  [1, H, W]  — ground-truth next state u(t_phys + Δt).
        """
        key, t_idx = self._index[idx]
        traj = self._load_trajectory(key)   # [T, 1, H, W]

        u_current = traj[t_idx]             # [1, H, W] — state at physical time t
        u_next    = traj[t_idx + 1]         # [1, H, W] — state at t + Δt  (TARGET)

        C = u_current

        if self.append_physical_time:
            # ── Physical PDE time ──────────────────────────────────────────────
            # This is grid/t[t_idx], a real-world timestamp (e.g. 0.5 s).
            # Broadcast to [1, H, W] so it slots in alongside spatial channels.
            #
            # Do NOT confuse with flow-matching τ ∈ [0,1], which the
            # FlowMatchingProcessor samples separately and passes to the model
            # as sample["x"]["t"].  That τ parameterises the generative path;
            # this t_phys tells the model where it is in the physical simulation.
            t_phys = self._t_grid[t_idx].item()
            t_map  = torch.full_like(u_current, t_phys)     # [1, H, W]
            C = torch.cat([C, t_map], dim=0)                 # [2, H, W]

        if self.append_coords:
            C = torch.cat([C, self._x_coord, self._y_coord], dim=0)  # [4, H, W]

        return C, u_next

    def _make_x0(self, idx: int, X_target: torch.Tensor) -> torch.Tensor:
        """Physical initial condition for flow matching: u(t_phys).

        For the SWE (time-dependent PDE) the flow-matching path is:
            X_0 = u(t)   →   X_1 = u(t + Δt)

        The "noise" distribution is the actual prior physical state, not a
        Gaussian cloud.  This means the model only has to learn a small,
        physically-structured correction rather than generating a solution
        from scratch — which is the right inductive bias for PDE time-stepping.

        Contrast with Darcy (static PDE):
            X_0 = zeros  →  X_1 = pressure field
        There the flow genuinely generates a solution; there is no prior state.

        Note: u(t) is channel 0 of C (u_current in _fetch_data_pair), so the
        values returned here exactly match C[0:1].  They are kept as a separate
        tensor because the base class and FlowMatchingProcessor treat "x_0" and
        the conditioning "x" as independent inputs.
        """
        key, t_idx = self._index[idx]
        traj = self._load_trajectory(key)   # [T, 1, H, W]
        return traj[t_idx]                  # [1, H, W] — u(t_phys)
