"""Tri-modal, token-preserving DecAlign for TurboVLA.

Core decomposition/alignment follows TACO-Group/DecAlign (MIT).  The deliberate
TurboVLA deviations are documented in ``DECALIGN_INTEGRATION.md`` and marked at
the relevant readout and masking sites below.
"""

from __future__ import annotations

import torch
from torch import nn

from .components import CrossModalTransformer, TokenFusion, masked_mean, zero_invalid
from .configuration import DecAlignConfig
from .objectives import MODALITIES, DecAlignObjectives
from .outputs import DecAlignLosses, DecAlignOutput


class TriModalDecAlign(nn.Module):
    def __init__(self, hidden_dim: int, config: DecAlignConfig) -> None:
        super().__init__()
        config.validate(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = config

        # Paper-faithful split: unique encoders do not share weights, while one
        # common encoder is shared by language, proprioception, and vision.
        self.unique_encoders = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in MODALITIES}
        )
        self.common_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transformers = nn.ModuleDict(
            {
                name: CrossModalTransformer(
                    hidden_dim=hidden_dim,
                    num_heads=config.num_heads,
                    num_layers=config.num_layers,
                    feedforward_multiplier=config.feedforward_multiplier,
                    attention_dropout=config.attention_dropout,
                    relu_dropout=config.relu_dropout,
                    residual_dropout=config.residual_dropout,
                    embedding_dropout=config.embedding_dropout,
                )
                for name in MODALITIES
            }
        )
        self.token_fusions = nn.ModuleDict(
            {name: TokenFusion(hidden_dim, config.fusion_dropout) for name in MODALITIES}
        )
        self.objectives = DecAlignObjectives(hidden_dim, config)

    def _validate_inputs(
        self,
        tokens: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> None:
        batch_size = tokens["language"].shape[0]
        for name in MODALITIES:
            value = tokens[name]
            mask = masks[name]
            if value.ndim != 3 or value.shape[0] != batch_size or value.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"{name} tokens must be [B,N,{self.hidden_dim}] with a shared batch size, got {tuple(value.shape)}"
                )
            if mask.shape != value.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(
                    f"{name} valid mask must be bool {tuple(value.shape[:2])}, "
                    f"got {mask.dtype} {tuple(mask.shape)}"
                )
            if not torch.all(mask.any(dim=1)):
                raise ValueError(f"every sample must contain at least one valid {name} token")

    @staticmethod
    def _context_for(
        target: str,
        unique: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve the original language/audio/vision ordering, with proprio
        # occupying the original audio slot.
        context_names = [name for name in MODALITIES if name != target]
        return (
            torch.cat([unique[name] for name in context_names], dim=1),
            torch.cat([masks[name] for name in context_names], dim=1),
        )

    def forward(
        self,
        language_tokens: torch.Tensor,
        proprio_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        language_valid_mask: torch.Tensor,
        proprio_valid_mask: torch.Tensor | None = None,
        vision_valid_mask: torch.Tensor | None = None,
        compute_objectives: bool = False,
    ) -> DecAlignOutput:
        if proprio_valid_mask is None:
            proprio_valid_mask = torch.ones(proprio_tokens.shape[:2], device=proprio_tokens.device, dtype=torch.bool)
        if vision_valid_mask is None:
            vision_valid_mask = torch.ones(vision_tokens.shape[:2], device=vision_tokens.device, dtype=torch.bool)
        tokens = {
            "language": language_tokens,
            "proprio": proprio_tokens,
            "vision": vision_tokens,
        }
        masks = {
            "language": language_valid_mask,
            "proprio": proprio_valid_mask,
            "vision": vision_valid_mask,
        }
        self._validate_inputs(tokens, masks)
        tokens = {name: zero_invalid(tokens[name], masks[name]) for name in MODALITIES}

        unique = {name: self.unique_encoders[name](tokens[name]) for name in MODALITIES}
        common = {name: self.common_encoder(tokens[name]) for name in MODALITIES}
        unique = {name: zero_invalid(unique[name], masks[name]) for name in MODALITIES}
        common = {name: zero_invalid(common[name], masks[name]) for name in MODALITIES}

        transformed_unique = {}
        for name in MODALITIES:
            context, context_mask = self._context_for(name, unique, masks)
            transformed_unique[name] = self.transformers[name](
                query=unique[name],
                context=context,
                query_valid_mask=masks[name],
                context_valid_mask=context_mask,
            )

        fused = {
            name: self.token_fusions[name](transformed_unique[name], common[name], masks[name])
            for name in MODALITIES
        }
        # TurboVLA adaptation: retain the baseline memory order and token count
        # (512 vision + 21 language + 2 proprio = 535 for LIBERO).
        action_memory = torch.cat([fused["vision"], fused["language"], fused["proprio"]], dim=1)

        heterogeneous_summary = [masked_mean(transformed_unique[name], masks[name]) for name in MODALITIES]
        homogeneous_summary = [masked_mean(common[name], masks[name]) for name in MODALITIES]
        summary_representation = torch.cat([*heterogeneous_summary, *homogeneous_summary], dim=1)

        losses = self.objectives(unique, common, masks) if compute_objectives else DecAlignLosses.zeros(action_memory)
        return DecAlignOutput(
            action_memory=action_memory,
            summary_representation=summary_representation,
            unique_tokens=unique,
            common_tokens=common,
            transformed_unique_tokens=transformed_unique,
            losses=losses,
        )
