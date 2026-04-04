import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def _train_test_trajectory_indices(
    n_trajectories: int, n_tr: int, n_te: int, seed: int
) -> Tuple[List[int], List[int]]:
    """Shuffle trajectory indices and split into disjoint train / test index lists."""
    n_tr = min(n_tr, n_trajectories)
    n_te = min(n_te, max(0, n_trajectories - n_tr))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_trajectories, generator=g).tolist()
    train_traj = perm[:n_tr]
    test_traj = perm[n_tr : n_tr + n_te]
    return train_traj, test_traj


def _train_test_sample_indices(
    n_samples: int, n_tr: int, n_te: int, seed: int
) -> Tuple[List[int], List[int]]:
    """Shuffle sample indices 0..N-1 and split into disjoint train / test index lists."""
    n_tr = min(n_tr, n_samples)
    n_te = min(n_te, max(0, n_samples - n_tr))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g).tolist()
    train_idx = perm[:n_tr]
    test_idx = perm[n_tr : n_tr + n_te]
    return train_idx, test_idx


class BaseSequentialDataset(Dataset):
    """
    Dataset of (x_t, x_{t+1}) pairs built only from the given trajectory indices.

    ``data_tensor`` has shape **(N, T, C, H, W)** where N is the number of trajectories.
    """

    def __init__(
        self,
        data_tensor: torch.Tensor,
        traj_indices: List[int],
        trajectory_metadata: Optional[Dict[int, Any]] = None,
        constant_params: Optional[torch.Tensor] = None,
    ):
        self.X, self.Y, self.traj_ids = self._create_pairs(data_tensor, traj_indices)
        self.trajectory_metadata = trajectory_metadata
        self.constant_params = constant_params

    @staticmethod
    def _create_pairs(
        data: torch.Tensor, traj_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        X_list, Y_list, ids = [], [], []
        for tid in traj_indices:
            traj = data[tid]
            if traj.shape[0] < 2:
                continue
            X_list.append(traj[:-1])
            Y_list.append(traj[1:])
            tp = traj.shape[0] - 1
            ids.extend([tid] * tp)
        if not X_list:
            raise ValueError(
                "Trajectory split produced no (t, t+1) pairs; ensure T>=2 per trajectory."
            )
        X = torch.cat(X_list, dim=0)
        Y = torch.cat(Y_list, dim=0)
        return X, Y, ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        out: Dict[str, Any] = {"x": self.X[idx], "y": self.Y[idx]}
        if self.constant_params is not None:
            out["params"] = self.constant_params.clone()
        if self.trajectory_metadata is not None:
            tid = self.traj_ids[idx]
            if tid in self.trajectory_metadata:
                out["metadata"] = self.trajectory_metadata[tid]
        return out


class BaseDataModule(abc.ABC):
    def __init__(
        self,
        name: str,
        data_dir: str,
        filename: Optional[str],
        batch_size: int,
        n_tr: int,
        n_te: int,
        seed: int = 42,
        trajectory_metadata: Optional[Dict[int, Any]] = None,
        constant_params: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.name = name
        self.batch_size = batch_size
        self.seed = seed
        self.trajectory_metadata = trajectory_metadata
        self.constant_params = constant_params
        self.cfg = DictConfig(
            {
                "root_dir": data_dir,
                "filename": filename,
                "n_tr": n_tr,
                "n_te": n_te,
            }
        )

    def get_dataloaders(self):
        """Load data, split trajectories (not flattened pairs), return train/test loaders."""
        raw = self.read_data()
        processed = self.preprocess(raw)
        if processed.dim() != 5:
            raise ValueError(
                f"Expected preprocess output of shape (N, T, C, H, W); got {tuple(processed.shape)}"
            )
        n_traj = processed.shape[0]
        train_traj, test_traj = _train_test_trajectory_indices(
            n_traj, int(self.cfg.n_tr), int(self.cfg.n_te), self.seed
        )
        data_tr = BaseSequentialDataset(
            processed,
            train_traj,
            self.trajectory_metadata,
            constant_params=self.constant_params,
        )
        data_te = BaseSequentialDataset(
            processed,
            test_traj,
            self.trajectory_metadata,
            constant_params=self.constant_params,
        )
        train_loader = DataLoader(data_tr, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(data_te, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_testing_data(self, n_samples=None, data_loader=True):
        """
        Test split uses the **same** trajectory-level shuffle as ``get_dataloaders`` (same seed).

        ``n_samples`` caps the number of **pair** samples when using a Subset (optional).
        """
        raw = self.read_data()
        processed = self.preprocess(raw)
        n_traj = processed.shape[0]
        _, test_traj = _train_test_trajectory_indices(
            n_traj, int(self.cfg.n_tr), int(self.cfg.n_te), self.seed
        )
        test_dataset = BaseSequentialDataset(
            processed,
            test_traj,
            self.trajectory_metadata,
            constant_params=self.constant_params,
        )

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

    def read_data(self) -> torch.Tensor:
        """Load raw tensor(s). Multiple ``*.npy`` / ``*.pt`` files are concatenated along the last axis (NS layout)."""
        root = Path(self.cfg.root_dir)
        filename = getattr(self.cfg, "filename", None)
        if filename:
            paths = [root / filename]
        else:
            paths = sorted(root.glob("*.npy")) + sorted(root.glob("*.pt"))
        if not paths:
            raise ValueError(f"No data files found under {root}")

        if len(paths) == 1:
            return self._load_tensor_from_path(paths[0])

        chunks = [self._load_tensor_from_path(p) for p in paths]
        # Navier-Stokes numpy layout is [T, H, W, N]; concatenate trajectories on last dim.
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def load_hdf5_arrays(
        path: Path,
        keys: Sequence[str],
        dtype: np.dtype = np.float32,
    ) -> Dict[str, torch.Tensor]:
        """Read named HDF5 datasets from a single file into CPU float tensors."""
        path = Path(path)
        out: Dict[str, torch.Tensor] = {}
        with h5py.File(path, "r") as f:
            available = list(f.keys())
            for key in keys:
                if key not in f:
                    raise KeyError(
                        f"Dataset {key!r} not in {path}; available top-level keys: {available}"
                    )
                arr = np.asarray(f[key][...], dtype=dtype)
                out[key] = torch.from_numpy(arr)
        return out

    @staticmethod
    def load_hdf5_array(
        path: Path, key: str, dtype: np.dtype = np.float32
    ) -> torch.Tensor:
        """Read one HDF5 dataset as a CPU float tensor."""
        return BaseDataModule.load_hdf5_arrays(path, (key,), dtype=dtype)[key]

    def _load_tensor_from_path(
        self,
        path: Path,
        *,
        hdf5_dataset_key: Optional[str] = None,
    ) -> torch.Tensor:
        path = Path(path)
        fp = path.as_posix().lower()
        if fp.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            return torch.from_numpy(np.asarray(arr)).float()
        if fp.endswith(".pt"):
            t = torch.load(path, weights_only=False)
            if not torch.is_tensor(t):
                t = torch.from_numpy(np.asarray(t)).float()
            return t.float()
        if fp.endswith(".h5") or fp.endswith(".hdf5"):
            if not hdf5_dataset_key:
                raise ValueError(
                    f"HDF5 file {path} requires a dataset name; pass "
                    f"hdf5_dataset_key=... or use load_hdf5_arrays(path, keys)."
                )
            return self.load_hdf5_array(path, hdf5_dataset_key)
        raise ValueError(f"Unsupported file format: {path}")

    @abc.abstractmethod
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Return shape [N, T, C, H, W]."""
        pass
