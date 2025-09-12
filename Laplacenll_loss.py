import torch, torch.nn.functional as F

def censored_laplace_nll(y_true, y_pred, lo=-1.0, hi=1.0, reduction="mean", eps=1e-6):
    mu, s = y_pred[..., 0], y_pred[..., 1]
    b = F.softplus(s) + 1e-6  # scale > 0

    y = y_true
    interior = (y > lo) & (y < hi)
    at_lo    = (y <= lo)
    at_hi    = (y >= hi)

    # Interior: standard Laplace NLL = log(2b)+|y-mu|/b
    nll_int = torch.log(2*b) + (y - mu).abs()/b

    # CDF for Laplace (stable pieces)
    # F(y) = 0.5*exp((y-mu)/b) for y<mu; 1 - 0.5*exp(-(y-mu)/b) for y>=mu
    z_lo = (lo - mu)/b
    z_hi = (hi - mu)/b
    # log CDF at lo
    logF_lo  = torch.where(lo < mu,  torch.log(torch.clamp(0.5*torch.exp(z_lo), min=eps)),
                                     torch.log1p(-0.5*torch.exp(-z_lo)))  # log(1 - 0.5 e^{-z})
    # log S at hi = log(1 - F(hi))
    logS_hi  = torch.where(hi < mu,  torch.log1p(-torch.clamp(0.5*torch.exp((hi-mu)/b), max=1-eps)),
                                     torch.log(torch.clamp(0.5*torch.exp(-(hi-mu)/b), min=eps)))

    # NLL for censored points
    nll_lo = -logF_lo
    nll_hi = -logS_hi

    nll = torch.zeros_like(y, dtype=y.dtype)
    nll[interior] = nll_int[interior]
    nll[at_lo]    = nll_lo[at_lo]
    nll[at_hi]    = nll_hi[at_hi]

    if reduction == "mean": return nll.mean()
    if reduction == "sum":  return nll.sum()
    return nll
