import torch
import torch.nn as nn


class KoPE(nn.Module):
    """Koopman Patch Embedding (Sec. 3.4): rolls the last encoder state forward, z_{N+i} = K^{i-1} z_N."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.M      = config.prediction_length // config.patch_size
        D           = config.embedding_dim

        # Orthogonal init. The bias is unused; dropping it would shift the seeded init of later modules.
        K_init  = torch.randn(D, D)
        U, _, V = torch.svd(K_init)
        self.K  = nn.Linear(D, D, bias=True)
        self.K.weight.data = torch.mm(U, V.t())

        self.input_norm  = nn.LayerNorm(D)
        self.output_norm = nn.LayerNorm(D)

        # Cached powers of K (populated at inference).
        self._K_powers        = None
        self._cache_signature = None

    def _parameter_signature(self):
        # Loading weights and moving the module must invalidate the powers cache.
        weight = self.K.weight
        return (weight._version, weight.data_ptr(), weight.device, weight.dtype)

    def _powers(self):
        """Stacked K^0 .. K^(M-1), shape (M, D, D), with K scaled so that ||K||_2 <= 1."""
        K = self.K.weight
        K = K / torch.linalg.matrix_norm(K, ord=2).clamp(min=1.0)
        K_powers = [torch.eye(K.shape[0], device=K.device, dtype=K.dtype)]
        for _ in range(self.M - 1):
            K_powers.append(K_powers[-1] @ K)
        return torch.stack(K_powers)

    def forward(self, z):
        """z: (B * V, N, D) encoder states. Returns the latent trajectory (B * V, M, D)."""
        z_N = self.input_norm(z[:, -1, :])

        if self.training:
            self._cache_signature = None
            K_powers = self._powers()
        else:
            signature = self._parameter_signature()
            if self._cache_signature != signature:
                with torch.no_grad():
                    self._K_powers = self._powers()
                self._cache_signature = signature
            K_powers = self._K_powers

        return self.output_norm(torch.einsum("bd,mde->bme", z_N, K_powers))
