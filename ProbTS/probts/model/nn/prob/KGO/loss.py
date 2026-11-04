import torch


CRPS_ENERGY_WEIGHT   = 0.5
CRPS_QUANTILE_WEIGHT = 0.5


def crps_energy_loss(samples, target):
    """Energy form of CRPS, E|X - y| - 0.5 E|X - X'|, with X' a shuffle of the samples.
    samples: (B, J, H, V); target: (B, H, V)."""
    J         = samples.shape[1]
    perm      = torch.randperm(J, device=samples.device)
    accuracy  = (samples - target.unsqueeze(1)).abs().mean(dim=1)
    repulsion = (samples - samples[:, perm]).abs().mean(dim=1)

    return (accuracy - 0.5 * repulsion).mean()


def crps_quantile_loss(samples, target, num_quantiles=9):
    """Pinball loss over the quantiles q = 1/10, ..., 9/10, normalized by sum |y| like the ProbTS CRPS metric."""
    quantiles = torch.arange(
        1, num_quantiles + 1, device=samples.device, dtype=samples.dtype
    ) / (num_quantiles + 1)

    q_forecasts = torch.quantile(samples, quantiles, dim=1)   # (Q, B, H, V)
    q           = quantiles[:, None, None, None]              # (Q, 1, 1, 1)
    errors      = target.unsqueeze(0) - q_forecasts           # (Q, B, H, V)
    pinball     = torch.where(errors >= 0, q * errors, (q - 1) * errors)

    abs_target = target.abs().sum() + 1e-8
    return (2 * pinball.sum() / abs_target) / len(quantiles)


def crps_loss(samples, target):
    """Training objective, Eq. (6). samples: (B, J, H, V) denormalized; target: (B, H, V)."""
    crps_e = crps_energy_loss(samples, target)
    crps_q = crps_quantile_loss(samples, target)
    return CRPS_ENERGY_WEIGHT * crps_e + CRPS_QUANTILE_WEIGHT * crps_q
