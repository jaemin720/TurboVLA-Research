from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecAlignConfig:
    """Architecture settings for the TurboVLA DecAlign adaptation.

    The first implementation intentionally exposes only the token-preserving
    readout.  A summary-token readout can later consume the same core outputs
    without changing the decomposition or alignment objectives.
    """

    enabled: bool = False
    representation_mode: str = "token_preserving"
    num_layers: int = 4
    num_heads: int = 8
    feedforward_multiplier: int = 4
    num_prototypes: int = 7
    gmm_em_iters: int = 5
    gmm_var_floor: float = 1e-4
    ot_regularization: float = 0.1
    ot_num_iters: int = 50
    pde_dim: int = 64
    mmd_bandwidth: float = 1.0
    attention_dropout: float = 0.0
    relu_dropout: float = 0.1
    residual_dropout: float = 0.05
    embedding_dropout: float = 0.1
    fusion_dropout: float = 0.0

    def validate(self, hidden_dim: int) -> None:
        if self.representation_mode != "token_preserving":
            raise ValueError("only decalign.representation_mode='token_preserving' is currently implemented")
        if self.num_layers < 1:
            raise ValueError("decalign.num_layers must be positive")
        if self.num_heads < 1 or hidden_dim % self.num_heads != 0:
            raise ValueError("TurboVLA hidden_dim must be divisible by decalign.num_heads")
        if self.feedforward_multiplier < 1:
            raise ValueError("decalign.feedforward_multiplier must be positive")
        if self.num_prototypes < 1 or self.gmm_em_iters < 1 or self.ot_num_iters < 1:
            raise ValueError("DecAlign prototype and iteration counts must be positive")
        if self.gmm_var_floor <= 0 or self.ot_regularization <= 0 or self.mmd_bandwidth <= 0:
            raise ValueError("DecAlign variance, OT regularization, and MMD bandwidth must be positive")
        if self.pde_dim < 1:
            raise ValueError("decalign.pde_dim must be positive")
