"""Neural components used by the TurboVLA DecAlign core.

The decomposition and modality-specific heterogeneous Transformers are adapted
from TACO-Group/DecAlign (MIT).  TurboVLA adaptation: the attention is
batch-first and padding-aware because language, proprioception, and vision have
different sequence lengths and only language contains padded tokens.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def zero_invalid(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


class CrossModalEncoderLayer(nn.Module):
    """Post-norm cross-attention layer matching DecAlign's layer ordering."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_multiplier: int,
        attention_dropout: float,
        relu_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * feedforward_multiplier)
        self.fc2 = nn.Linear(hidden_dim * feedforward_multiplier, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.relu_dropout = nn.Dropout(relu_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_valid_mask: torch.Tensor,
        context_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query=query,
            key=context,
            value=context,
            key_padding_mask=~context_valid_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(query + self.residual_dropout(attended))
        residual = hidden
        hidden = self.fc2(self.relu_dropout(F.relu(self.fc1(hidden))))
        hidden = self.final_norm(residual + self.residual_dropout(hidden))
        return zero_invalid(hidden, query_valid_mask)


class CrossModalTransformer(nn.Module):
    """Modality-specific Transformer over unique DecAlign features."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_multiplier: int,
        attention_dropout: float,
        relu_dropout: float,
        residual_dropout: float,
        embedding_dropout: float,
    ) -> None:
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.layers = nn.ModuleList(
            [
                CrossModalEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    feedforward_multiplier=feedforward_multiplier,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_valid_mask: torch.Tensor,
        context_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embedding_dropout(query * self.scale)
        scaled_context = self.embedding_dropout(context * self.scale)
        hidden = zero_invalid(hidden, query_valid_mask)
        scaled_context = zero_invalid(scaled_context, context_valid_mask)
        for layer in self.layers:
            hidden = layer(hidden, scaled_context, query_valid_mask, context_valid_mask)
        return hidden


class TokenFusion(nn.Module):
    """Fuse unique and common features without changing modality token count."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        transformed_unique: torch.Tensor,
        common: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # TurboVLA adaptation: the original DecAlign pools these tensors into
        # two vectors per modality.  Channel fusion retains all 535 tokens.
        fused = self.norm(self.dropout(self.projection(torch.cat([transformed_unique, common], dim=-1))))
        return zero_invalid(fused, valid_mask)
