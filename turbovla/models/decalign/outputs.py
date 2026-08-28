from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DecAlignLosses:
    decoupling: torch.Tensor
    heterogeneous: torch.Tensor
    homogeneous: torch.Tensor
    prototype_ot: torch.Tensor
    local_prototype: torch.Tensor
    distribution_statistics: torch.Tensor
    mmd: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "DecAlignLosses":
        zero = reference.new_zeros(())
        return cls(zero, zero, zero, zero, zero, zero, zero)


@dataclass
class DecAlignOutput:
    # TurboVLA adaptation: preserve the baseline memory layout
    # [vision tokens, language tokens, proprio tokens].
    action_memory: torch.Tensor
    summary_representation: torch.Tensor
    unique_tokens: dict[str, torch.Tensor]
    common_tokens: dict[str, torch.Tensor]
    transformed_unique_tokens: dict[str, torch.Tensor]
    losses: DecAlignLosses
