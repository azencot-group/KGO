import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # num_layers counts the Linear layers; there is always at least one hidden layer.
        layers, dim = [], in_dim
        for _ in range(max(num_layers - 1, 1)):
            layers += [nn.Linear(dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(dropout)]
            dim     = h_dim
        layers.append(nn.Linear(h_dim, out_dim))

        self.network        = nn.Sequential(*layers)
        self.residual_layer = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.network(x) + self.residual_layer(x)
