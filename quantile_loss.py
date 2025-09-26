import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileLoss(nn.Module):
    """
    Quantile (pinball) or quantile-Huber loss with optional non-crossing penalty.

    Args:
        taus: iterable of quantiles in ascending order (e.g., (0.2, 0.8)).
        kappa: >0 enables quantile-Huber smoothing (≈ 0.05–0.2 on label scale);
               0 or None → pure pinball.
        reduction: 'mean' | 'sum' | 'none' (applied after summing across taus).
        lambda_non_cross: penalty weight for max(Q_low - Q_high, 0) if len(taus) >= 2.
    """
    def __init__(self, taus=(0.2, 0.8), kappa: float | None = None,
                 reduction: str = "mean", lambda_non_cross: float = 0.0):
        super().__init__()
        taus = torch.as_tensor(taus, dtype=torch.float32).view(1, -1)
        if taus.numel() == 0:
            raise ValueError("taus must be non-empty.")
        if not torch.all(taus.flatten()[1:] >= taus.flatten()[:-1]):
            raise ValueError("taus must be in ascending order.")
        self.register_buffer("taus", taus)
        self.kappa = float(kappa) if (kappa is not None and kappa > 0) else 0.0
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean' | 'sum' | 'none'.")
        self.reduction = reduction
        self.lambda_non_cross = float(lambda_non_cross)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input : [B, T] predicted quantiles (T = len(taus), same order).
        target: [B]    true values.
        """
        if input.ndim != 2 or input.size(1) != self.taus.size(1):
            raise ValueError(f"input must be [B, {self.taus.size(1)}], got {tuple(input.shape)}")
        if target.ndim != 1 or target.size(0) != input.size(0):
            raise ValueError(f"target must be [B], got {tuple(target.shape)}")

        y = target.to(dtype=input.dtype, device=input.device).unsqueeze(-1)  # [B,1]
        q = input                                                             # [B,T]
        taus = self.taus.to(dtype=input.dtype, device=input.device)           # [1,T]
        u = y - q                                                             # [B,T]

        if self.kappa > 0.0:
            # Quantile-Huber (tilted Huber)
            absu = u.abs()
            quad = 0.5 * (absu ** 2) / self.kappa
            lin  = absu - 0.5 * self.kappa
            hub  = torch.where(absu <= self.kappa, quad, lin)
            w    = torch.where(u >= 0, taus, 1.0 - taus)  # asymmetry
            loss_q = w * hub
        else:
            # Pinball
            loss_q = torch.maximum(taus * u, (taus - 1.0) * u)

        loss = loss_q.sum(dim=1)  # sum over quantiles → [B]

        # Optional non-crossing penalty: enforce Q_low <= Q_high
        if self.lambda_non_cross > 0.0 and q.size(1) >= 2:
            penalty = F.relu(q[:, 0] - q[:, -1])          # [B]
            loss = loss + self.lambda_non_cross * penalty

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # 'none'

# --- Example usage ---
# model outputs [Q0.2, Q0.8] per sample
# crit = QuantileLoss(taus=(0.2, 0.8), kappa=0.1, reduction="mean", lambda_non_cross=1e-2)
# loss = crit(y_pred, y_true)




import math
import torch
import torch.nn as nn

class QuantileToScore(nn.Module):
    """
    Collapse [Q0.2, Q0.8] -> single scalar score.

    modes:
      - 'mean'   : mu
      - 'ce'     : certainty-equivalent mu - kappa*b*sign(mu)   (recommended)
      - 'sharpe' : mu / (b + eps)
      - 'bound'  : q20 if mu>=0 else q80
      - 'prob'   : 2*P(Y>0)-1 under Laplace(mu,b) in [-1,1]
    """
    def __init__(self, mode='ce', kappa=0.75, eps=1e-6):
        super().__init__()
        assert mode in {'mean','ce','sharpe','bound','prob'}
        self.mode  = mode
        self.kappa = float(kappa)
        self.eps   = float(eps)
        # constant: 2*ln(2.5) ≈ 1.83258
        self.register_buffer('two_log_2p5', torch.tensor(2.0*math.log(2.5), dtype=torch.float32))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [B, 2] where q[:,0]=Q0.2, q[:,1]=Q0.8 (must satisfy Q0.2 <= Q0.8)
        returns: [B] score
        """
        if q.ndim != 2 or q.size(1) != 2:
            raise ValueError(f"expected [B,2] for [Q0.2,Q0.8], got {tuple(q.shape)}")

        q20, q80 = q[:, 0], q[:, 1]
        mu = 0.5*(q20 + q80)
        spread = (q80 - q20).clamp_min(0.0)
        b = spread / (self.two_log_2p5 + 1e-12)  # Laplace scale estimate

        if self.mode == 'mean':
            return mu

        if self.mode == 'ce':
            # certainty-equivalent: shrink toward 0 by kappa*b
            return mu - self.kappa * b * torch.sign(mu)

        if self.mode == 'sharpe':
            return mu / (b + self.eps)

        if self.mode == 'bound':
            # adverse-side bound as scalar prediction
            return torch.where(mu >= 0, q20, q80)

        if self.mode == 'prob':
            # signed probability (in [-1,1]) under Laplace(mu,b)
            b_safe = b + self.eps
            pos = mu > 0
            ppos = torch.empty_like(mu)
            ppos[pos]  = 1.0 - 0.5*torch.exp(-mu[pos]/b_safe[pos])
            ppos[~pos] = 0.5*torch.exp( mu[~pos]/b_safe[~pos])
            return 2.0*ppos - 1.0

