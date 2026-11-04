# KoopmanLift and KoopmanRecon adapt the AdaLN residual block and final layer of the Sundial
# flow-matching head (https://github.com/thuml/Sundial), which follow MAR and DiT.

import torch
import torch.nn as nn


# Probability of training from pure noise (tau = 0), where one-step sampling starts.
TAU0_PROB = 0.5


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class KoopmanLift(nn.Module):
    def __init__(self, patch_size: int, cond_dim: int, latent_dim: int):
        super().__init__()
        self.cond_projection  = nn.Linear(cond_dim, latent_dim)
        self.input_projection = nn.Linear(patch_size, latent_dim)
        self.norm             = nn.LayerNorm(latent_dim, eps=1e-6)
        self.mlp              = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=True),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 3 * latent_dim, bias=True),
        )

    def condition(self, c):
        """c: (BM, V, D) KoPE latents. Returns (c_proj, shift, scale, gate), each (BM, V, d)."""
        c_proj             = self.cond_projection(c)
        shift, scale, gate = self.adaLN_modulation(c_proj).chunk(3, dim=-1)
        return c_proj, shift, scale, gate

    def forward(self, x, cond):
        """x: (BM * n, V, P), n samples per conditioning row. Returns (BM * n, V, d)."""
        _, shift, scale, gate = cond
        BM = shift.shape[0]
        z  = self.input_projection(x)
        z  = z.reshape(BM, -1, *z.shape[1:])                                # (BM, n, V, d)
        h  = modulate(self.norm(z), shift.unsqueeze(1), scale.unsqueeze(1))
        z  = z + gate.unsqueeze(1) * self.mlp(h)
        return z.flatten(0, 1)


class KoopmanRecon(nn.Module):
    def __init__(self, patch_size: int, latent_dim: int):
        super().__init__()
        self.decode           = nn.Linear(latent_dim, patch_size)
        self.norm             = nn.LayerNorm(latent_dim, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim, bias=True),
        )

    def condition(self, c_proj):
        """c_proj: (BM, V, d) from KoopmanLift.condition. Returns (shift, scale)."""
        shift, scale = self.adaLN_modulation(c_proj).chunk(2, dim=-1)
        return shift, scale

    def forward(self, z, cond):
        """z: (BM * n, V, d). Returns (BM * n, V, P)."""
        shift, scale = cond
        BM = shift.shape[0]
        z  = z.reshape(BM, -1, *z.shape[1:])                                # (BM, n, V, d)
        z  = modulate(self.norm(z), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.decode(z).flatten(0, 1)


class KoopmanTransport(nn.Module):
    """Eq. (4): z^(1) = e^{dtau A} z^(tau) + dtau b with dtau = 1 - tau. Training snaps dtau to `num_bins`
    values, one matrix exponential per bin (App. B.2.3); inference (dtau = 1) caches e^A."""

    def __init__(self, latent_dim, num_bins=32):
        super().__init__()
        self.num_bins = num_bins
        self.A        = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.b        = nn.Parameter(torch.zeros(latent_dim))

        self._cached_expA     = None
        self._cache_signature = None

    def _parameter_signature(self):
        # Loading weights and moving the module must invalidate the cached e^A.
        return tuple((p._version, p.data_ptr(), p.device, p.dtype) for p in (self.A, self.b))

    def forward(self, z, dtau=None):
        """z: (n, V, d); dtau: (n,) remaining transport time, or None for one-step inference."""
        if dtau is None:
            signature = self._parameter_signature()
            if self._cache_signature != signature:
                with torch.no_grad():
                    self._cached_expA = torch.matrix_exp(self.A)
                self._cache_signature = signature
            return z @ self._cached_expA + self.b

        self._cache_signature = None
        num_bins = self.num_bins
        bins     = torch.linspace(0.0, 1.0, num_bins, device=dtau.device)        # (num_bins,)

        # Snap each dtau to its nearest bin; the clamp guards against float round-off.
        bin_idx  = torch.round(dtau * (num_bins - 1)).long().clamp(0, num_bins - 1)  # (n,)
        bin_dtau = bins[bin_idx]                                                  # (n,)

        expA = torch.matrix_exp(bins[:, None, None] * self.A)                     # (num_bins, d, d)
        return torch.bmm(z, expA[bin_idx]) + bin_dtau[:, None, None] * self.b


class KoFM(nn.Module):
    """Koopman Flow Matching (Sec. 3.5): lift x^(tau) conditioned on the KoPE latent, transport it
    to tau = 1 in closed form, and decode. Rows are ordered (batch, patch, sample)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        P, D, d     = config.patch_size, config.embedding_dim, config.kofm_dim

        self.lift      = KoopmanLift(P, D, d)
        self.recon     = KoopmanRecon(P, d)
        self.transport = KoopmanTransport(d)

    def condition(self, c):
        """Noise-independent conditioning, computed once and shared by all samples. c: (BM, V, D)."""
        lift_cond = self.lift.condition(c)
        return lift_cond, self.recon.condition(lift_cond[0])

    def sample(self, cond, n):
        """One-step sampling from pure noise (tau = 0): n samples per row. Returns (BM * n, V, P)."""
        lift_cond, recon_cond = cond
        c_proj = lift_cond[0]
        BM, V  = c_proj.shape[:2]
        noise  = torch.randn(BM * n, V, self.config.patch_size, device=c_proj.device)
        z_1    = self.transport(self.lift(noise, lift_cond))
        return self.recon(z_1, recon_cond)

    def forward(self, c, y):
        """Training pass, J samples per row. c: (BM, V, D); y: (BM, V, P) target patches.
        Returns (BM * J, V, P)."""
        lift_cond, recon_cond = self.condition(c)
        y     = y.repeat_interleave(self.config.num_train_samples, dim=0)
        eps   = torch.randn_like(y)
        tau   = torch.rand(y.shape[0], device=y.device)
        tau   = tau.masked_fill(torch.rand_like(tau) < TAU0_PROB, 0.0)
        x_tau = (1 - tau[:, None, None]) * eps + tau[:, None, None] * y
        z_1   = self.transport(self.lift(x_tau, lift_cond), 1 - tau)
        return self.recon(z_1, recon_cond)
