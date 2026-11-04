# Adapted from https://github.com/ts-kim/RevIN, trimmed to the paths KGO uses.

from contextlib import contextmanager

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps          = eps
        self.affine       = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        # x: (B, L, V)
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean  = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        return x * self.stdev + self.mean

    @contextmanager
    def expanded(self, k):
        """Repeat the stored statistics k times along the batch dim, so k samples per series
        can be denormalized in one call."""
        mean, stdev = self.mean, self.stdev
        self.mean   = mean.repeat_interleave(k, dim=0)
        self.stdev  = stdev.repeat_interleave(k, dim=0)
        try:
            yield
        finally:
            self.mean, self.stdev = mean, stdev
