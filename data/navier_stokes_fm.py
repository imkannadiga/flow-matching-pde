from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data.base import BaseDataModule, _train_test_trajectory_indices


class FlowMatchingSequentialDataset(Dataset):
    """
    (x_t, x_{t+1}) pairs plus a normalized **physical** time index along each trajectory.

    Built only from the given trajectory indices so train/test stay trajectory-disjoint.
    """

    def __init__(
        self,
        data_tensor: torch.Tensor,
        traj_indices: List[int],
        trajectory_metadata: Optional[Dict[int, Any]] = None,
    ):
        self.X, self.Y, self.T, self.traj_ids = self._create_pairs(
            data_tensor, traj_indices
        )
        self.trajectory_metadata = trajectory_metadata

    @staticmethod
    def _create_pairs(
        data: torch.Tensor, traj_indices: List[int]
    ):
        X_list, Y_list, T_list, ids = [], [], [], []
        for tid in traj_indices:
            traj = data[tid]
            if traj.shape[0] < 2:
                continue
            time_steps = traj.shape[0] - 1
            X_list.append(traj[:-1])
            Y_list.append(traj[1:])
            denom = max(time_steps, 1)
            t_vals = torch.linspace(0, 1 - 1 / denom, time_steps)
            T_list.append(t_vals)
            ids.extend([tid] * time_steps)
        if not X_list:
            raise ValueError(
                "Trajectory split produced no FM pairs; ensure T>=2 per trajectory."
            )
        X = torch.cat(X_list, dim=0)
        Y = torch.cat(Y_list, dim=0)
        Tcat = torch.cat(T_list, dim=0)
        return X, Y, Tcat, ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        out: Dict[str, Any] = {
            "x": self.X[idx],
            "y": self.Y[idx],
            "t": self.T[idx],
        }
        if self.trajectory_metadata is not None:
            tid = self.traj_ids[idx]
            if tid in self.trajectory_metadata:
                out["metadata"] = self.trajectory_metadata[tid]
        return out


class NSFMDataModule(BaseDataModule):
    """Navier-Stokes data for flow matching with per-sample physical time ``t``."""

    domain_bounds = ((0.0, 1.0), (0.0, 1.0))

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        data = data.permute(3, 0, 1, 2)
        data = data.unsqueeze(2)
        return data

    def get_dataloaders(self):
        raw = self.read_data()
        processed = self.preprocess(raw)
        n_traj = processed.shape[0]
        train_traj, test_traj = _train_test_trajectory_indices(
            n_traj, int(self.cfg.n_tr), int(self.cfg.n_te), self.seed
        )
        data_tr = FlowMatchingSequentialDataset(
            processed, train_traj, self.trajectory_metadata
        )
        data_te = FlowMatchingSequentialDataset(
            processed, test_traj, self.trajectory_metadata
        )
        train_loader = DataLoader(data_tr, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(data_te, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_testing_data(self, n_samples: int = None, data_loader: bool = True):
        raw = self.read_data()
        processed = self.preprocess(raw)
        n_traj = processed.shape[0]
        _, test_traj = _train_test_trajectory_indices(
            n_traj, int(self.cfg.n_tr), int(self.cfg.n_te), self.seed
        )
        test_dataset = FlowMatchingSequentialDataset(
            processed, test_traj, self.trajectory_metadata
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
