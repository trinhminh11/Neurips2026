from __future__ import annotations

import torch
from torch import nn


def _init_fc(
    layer: nn.Linear | nn.Sequential, initializer: str = "xavier_uniform"
) -> None:
    if isinstance(layer, nn.Linear):
        if initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Sequential):
        for sublayer in layer:
            if isinstance(sublayer, nn.Linear):
                if initializer == "xavier_uniform":
                    nn.init.xavier_uniform_(sublayer.weight)
                else:
                    nn.init.kaiming_normal_(sublayer.weight)
                sublayer.bias.data.zero_()
            else:
                return None
    else:
        return None


class TwoLayerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        _init_fc(self.net)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)
