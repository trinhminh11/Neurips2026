"""Trajectory-conditioned skill discriminator q_phi(z | H_t^k) via Transformer."""

from __future__ import annotations

import torch
from torch import nn

from common_net.attentions import MAB, ChainBlock, MHAConfig
from common_net.attentions.base_attn import ScaledDotProductAttention
from common_net.pos_embedding.sinusoidal_positonal import SinusoidalPositionalEmbedding


class TrajectoryDiscriminator(nn.Module):
    """
    Encodes a variable-length trajectory (B, T, n_states) with a [CLS] token,
    stacked MAB blocks, and a linear classifier to skill logits.
    """

    def __init__(
        self,
        n_states: int,
        n_skills: int,
        embed_dim: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.n_states = n_states
        self.embed_dim = embed_dim
        self.state_proj = nn.Linear(n_states, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        mha_config = MHAConfig(
            num_heads=num_heads,
            attn_cls=ScaledDotProductAttention,
            attn_kwargs={},
            positional_embedding_cls=SinusoidalPositionalEmbedding,
            positional_embedding_kwargs={},
            gated=False,
        )
        self.blocks = ChainBlock(
            num_repeats=num_layers,
            block_cls=MAB,
            embed_dim=embed_dim,
            d_ff=d_ff,
            mha_config=mha_config,
        )
        self.head = nn.Linear(embed_dim, n_skills)
        nn.init.xavier_uniform_(self.head.weight)
        self.head.bias.data.zero_()

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (B, T, n_states) consecutive states in segment H_t^k (or prefix).
        Returns:
            logits: (B, n_skills)
        """
        B, T, _ = trajectory.shape
        x = self.state_proj(trajectory)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)
        h = self.blocks(h, causal=False)
        cls_out = h[:, 0]
        return self.head(cls_out)
