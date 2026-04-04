from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .common import TwoLayerMLP, _init_fc


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        action_scale: float = 1,
        action_bias: float = 0,
        n_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = TwoLayerMLP(n_states, n_hidden)

        self.mu = nn.Linear(n_hidden, n_actions)
        self.log_std = nn.Linear(n_hidden, n_actions)

        _init_fc(self.mu)
        _init_fc(self.log_std)

    def forward(self, states: torch.Tensor) -> Normal:
        x: torch.Tensor = self.net(states)
        mu: torch.Tensor = self.mu(x)
        log_std: torch.Tensor = self.log_std(x).clamp(min=-20, max=2)
        std = log_std.exp()
        return Normal(mu, std)

    def sample_and_log_prob(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist: Normal = self(states)
        u = dist.rsample()
        actions = torch.tanh(u)
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)    # do not forget to add the correction for Tanh squashing
        log_prob = log_prob.sum(-1, keepdim=True)

        actions = (actions * self.action_scale) + self.action_bias

        return actions, log_prob
