from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


class SampleStore:
    """Randomly stores up to ``n_samples`` input/output pairs during evaluation.

    Uses reservoir sampling so the budget is respected without knowing the dataset
    size upfront. Each stored entry is a dict with keys:
      - ``condition``: physical input [C_cond, H, W]
      - ``pred``:      model prediction [C, H, W]
      - ``true``:      ground-truth target [C, H, W]
      - ``rollout``:   list of intermediate states (optional, see ``rollout_length``)

    Configurable via Hydra:

    .. code-block:: yaml

        sample_store:
          _target_: evaluation.sample_store.SampleStore
          n_samples: 16
          save_dir: ${hydra:runtime.output_dir}
          rollout_length: null    # null = final prediction only
          filename: samples.pt
    """

    def __init__(
        self,
        n_samples: int,
        save_dir: str,
        rollout_length: Optional[int] = None,
        filename: str = "samples.pt",
    ):
        """
        Parameters
        ----------
        n_samples : int
            Maximum number of individual samples to retain.
        save_dir : str
            Directory where ``filename`` will be written.
        rollout_length : int or None
            If set, store up to this many evenly-spaced intermediate states per sample.
            None means only the final prediction is stored.
        filename : str
            Output filename within ``save_dir``.
        """
        self.n_samples = int(n_samples)
        self.save_dir = Path(save_dir)
        self.rollout_length = rollout_length
        self.filename = filename
        self._reservoir: list[dict] = []
        self._seen = 0

    def reset(self) -> None:
        self._reservoir = []
        self._seen = 0

    def maybe_store(
        self,
        condition: Tensor,
        pred: Tensor,
        true: Tensor,
        rollout: Optional[list[Tensor]] = None,
    ) -> None:
        """Reservoir-sample individual items from a batch.

        Parameters
        ----------
        condition : Tensor [B, C_cond, H, W]
        pred      : Tensor [B, C, H, W]
        true      : Tensor [B, C, H, W]
        rollout   : list of Tensor [B, C, H, W], optional
            One tensor per integration step. Subsampled to ``rollout_length`` frames.
        """
        B = pred.shape[0]

        # Subsample rollout to configured length
        trimmed_rollout: Optional[list[Tensor]] = None
        if rollout is not None and self.rollout_length is not None and len(rollout) > 0:
            step = max(1, len(rollout) // self.rollout_length)
            trimmed_rollout = rollout[::step][: self.rollout_length]

        for i in range(B):
            self._seen += 1
            entry: dict = {
                "condition": condition[i].cpu(),
                "pred": pred[i].cpu(),
                "true": true[i].cpu(),
            }
            if trimmed_rollout is not None:
                entry["rollout"] = [r[i].cpu() for r in trimmed_rollout]

            if len(self._reservoir) < self.n_samples:
                self._reservoir.append(entry)
            else:
                j = random.randint(0, self._seen - 1)
                if j < self.n_samples:
                    self._reservoir[j] = entry

    def save(self) -> None:
        """Write stored samples to disk as a list of dicts."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._reservoir, self.save_dir / self.filename)

    def __len__(self) -> int:
        return len(self._reservoir)
