from __future__ import annotations

import torch
from common_net.common import get_activation
from torch import nn

from .common import _init_fc


class SpectralMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: None | list[int] = None,
        bias: bool = True,
        activation_fn: str = "relu",
    ):
        super(SpectralMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = []

        dims = [input_dim] + hidden_dims + [output_dim]

        layers = []

        for i in range(len(dims) - 1):
            layers.append(
                torch.nn.utils.spectral_norm(nn.Linear(dims[i], dims[i + 1], bias=bias))
            )
            if i < len(dims) - 2:
                layers.append(get_activation(activation_fn))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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

        self.net = SpectralMLP(k * n_states, n_skills, [hidden_dims] * 2)
        _init_fc(self.net)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        # trajectory: B, k, n_states

        return self.net(trajectory.reshape(-1, self.k * self.n_states))


class OneStateDiscriminator(nn.Module):
    def __init__(self, n_states: int, n_skills: int, hidden_dims: int = 1024) -> None:
        super().__init__()
        self.net = SpectralMLP(n_states, n_skills, [hidden_dims] * 2)
        _init_fc(self.net)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
