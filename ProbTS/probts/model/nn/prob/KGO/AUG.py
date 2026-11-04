import torch
import torch.nn as nn
import torch.nn.functional as F


class AUG(nn.Module):
    """Adaptive Uncertainty Gate (Sec. 3.6): per-variate, per-horizon weights of the stochastic (g_h)
    and deterministic (1 - g_h) streams, computed from the raw context."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        L           = config.context_length
        H           = config.prediction_length
        hidden_dim  = max(L // 4, 16)

        self.variate_encoder = nn.Sequential(
            nn.Linear(L, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.horizon_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, H * 2),
        )

        self.gate_temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """x: (B, L, V). Returns (B, 2, H, V)."""
        B, _, V = x.shape
        H       = self.config.prediction_length

        h      = self.variate_encoder(x.permute(0, 2, 1))
        logits = self.horizon_router(h).view(B, V, H, 2)

        # Gumbel-softmax with a learned temperature in training, plain softmax at inference.
        if self.training:
            temp  = torch.clamp(self.gate_temperature, min=0.5, max=2.0)
            gates = F.gumbel_softmax(logits, tau=temp, hard=False, dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)

        return gates.permute(0, 3, 2, 1)
