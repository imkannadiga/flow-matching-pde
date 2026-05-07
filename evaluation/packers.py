from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class ConditionPacker(ABC):
    """Builds the model input ``u`` from the evolving state and an optional physical condition.

    Swappable via Hydra config. The packer used at eval time must match the conditioning
    strategy used during training, since it determines the model's input channel count.
    """

    @abstractmethod
    def pack(self, state: Tensor, condition: Optional[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        state : Tensor
            Current evolving state [B, C_state, H, W].
        condition : Tensor or None
            Physical conditioning field [B, C_cond, H, W].

        Returns
        -------
        Tensor
            Packed model input ``u``.
        """


class NullPacker(ConditionPacker):
    """Passes state only — no physical conditioning.

    Use when the model was trained without concatenated conditioning (e.g. current default
    training setup, or unconditional generation). Also the natural default for time-invariant
    PDEs where conditioning is implicit in the noise initialisation.
    """

    def pack(self, state: Tensor, condition: Optional[Tensor]) -> Tensor:
        return state


class PhysicalPacker(ConditionPacker):
    """Concatenates the physical condition along the channel dim: ``u = cat([condition, state])``.

    Use for time-variant PDEs or when training was updated to include physical conditioning
    in the model input. Requires ``condition`` to be non-None.
    """

    def pack(self, state: Tensor, condition: Optional[Tensor]) -> Tensor:
        if condition is None:
            raise ValueError("PhysicalPacker requires a non-None condition tensor.")
        return torch.cat([condition, state], dim=1)
