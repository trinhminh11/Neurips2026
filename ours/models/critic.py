from __future__ import annotations

import torch
from torch import nn

from .common import TwoLayerMLP, _init_fc


class CriticNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, n_hidden: int = 1024) -> None:
        super().__init__()

        self.net = TwoLayerMLP(n_states + n_actions, n_hidden)
        self.q = nn.Linear(n_hidden, 1)
        _init_fc(self.q)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        x = self.net(x)
        return self.q(x)

    def soft_update(self, target: CriticNetwork, tau: float = 0.005) -> None:
        for target_param, param in zip(target.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
