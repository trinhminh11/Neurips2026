from __future__ import annotations

import torch
from torch import nn

from common_net.common import MLP

from .common import _init_fc


class MLPDiscriminator(nn.Module):
    """
    Encodes a variable-length trajectory (B, T, n_states) with a [CLS] token,
    stacked MAB blocks, and a linear classifier to skill logits.
    """

    def __init__(
        self,
        k: int,
        n_states: int,
        n_skills: int,
        hidden_dims: int = 1024,
    ) -> None:
        super().__init__()
        self.k = k
        self.n_states = n_states

        self.net = MLP(k * n_states, n_skills, [hidden_dims] * 2)
        _init_fc(self.net)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        # trajectory: B, k, n_states

        return self.net(trajectory.reshape(-1, self.k * self.n_states))


class OneStateDiscriminator(nn.Module):
    def __init__(self, n_states: int, n_skills: int, hidden_dims: int = 1024) -> None:
        super().__init__()
        self.net = MLP(n_states, n_skills, [hidden_dims] * 2)
        _init_fc(self.net)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
