import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.embedding = nn.Linear(config.patch_size, config.embedding_dim)

    def forward(self, x):
        """x: (B, L, V). Returns (B * V, N, D): N = L / P non-overlapping patches, embedded per variate."""
        P = self.config.patch_size
        x = rearrange(x, "B L V -> (B V) L")
        x = x.unfold(dimension=-1, size=P, step=P)
        return self.embedding(x)
