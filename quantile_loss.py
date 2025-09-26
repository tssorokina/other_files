import torch
import torch.nn.functional as F

def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    taus=(0.2, 0.8),
    kappa: float | None = None,        # None or 0 → pure pinball; >0 → quantile-Huber (smooth)
    reduction: str = "mean",
    lambda_non_cross: float = 0.0      # add >0 to penalize Q_low > Q_high
):
    """
    y_true: [B]
    y_pred: [B, len(taus)]  (column i = predicted quantile for taus[i])
    taus:   iterable of quantiles in ascending order (e.g., (0.2, 0.8))
    kappa:  Huber smoothing parameter for residuals (recommended ~0.1..0.2 of label scale)
    """
    if y_pred.ndim != 2 or y_pred.size(1) != len(taus):
        raise ValueError(f"y_pred should be [B, {len(taus)}], got {tuple(y_pred.shape)}")

    y = y_true.unsqueeze(-1)                 # [B, 1]
    q = y_pred                               # [B, T]
    taus_t = torch.tensor(taus, device=y.device, dtype=y.dtype).view(1, -1)  # [1, T]
    u = y - q                                # residuals

    if not kappa or kappa <= 0:
        # Pinball: max(τ*u, (τ-1)*u)
        loss = torch.maximum(taus_t * u, (taus_t - 1.0) * u)
    else:
        # Quantile-Huber (tilted Huber)
        absu = u.abs()
        quad = 0.5 * (absu ** 2) / kappa
        lin  = absu - 0.5 * kappa
        hub  = torch.where(absu <= kappa, quad, lin)
        w    = torch.where(u >= 0, taus_t, 1.0 - taus_t)  # asymmetry per side
        loss = w * hub

    # Non-crossing penalty: encourage Q_low <= Q_high
    if lambda_non_cross > 0 and len(taus) >= 2:
        q_low  = q[:, 0]
        q_high = q[:, -1]
        penalty = F.relu(q_low - q_high)  # [B]
        loss = loss.sum(dim=1) + lambda_non_cross * penalty
    else:
        loss = loss.sum(dim=1)            # sum over quantiles

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")
