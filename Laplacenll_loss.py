import torch
import torch.nn.functional as F

def laplace_nll(y_true: torch.Tensor,
                y_pred: torch.Tensor,
                reduction: str = "mean",
                min_scale: float = 1e-6,
                scale_param: str = "softplus"):
    """
    Laplace negative log-likelihood with heteroscedastic scale.

    Args:
        y_true: (...,)     target.
        y_pred: (..., 2)   model output where y_pred[...,0]=mu, y_pred[...,1]=s (unconstrained).
        reduction: 'mean' | 'sum' | 'none'.
        min_scale: small floor added to keep scale strictly positive.
        scale_param:
            - 'softplus' (default): b = softplus(s) + min_scale     <-- safest
            - 'exp':               b = exp(s) + min_scale
            - 'positive':          b = s + min_scale  (use only if network already outputs b>=0)

    Returns:
        loss tensor (scalar if reduction!='none').
    """
    if y_pred.size(-1) != 2:
        raise ValueError(f"y_pred last dim must be 2 (mu, s); got {y_pred.size(-1)}")

    mu = y_pred[..., 0]
    s  = y_pred[..., 1]

    if scale_param == "softplus":
        b = F.softplus(s) + min_scale
    elif scale_param == "exp":
        b = torch.exp(s) + min_scale
    elif scale_param == "positive":
        b = s + min_scale
    else:
        raise ValueError("scale_param must be one of {'softplus','exp','positive'}")

    # Laplace NLL: log(2b) + |y - mu|/b
    nll = torch.log(2.0 * b) + (y_true - mu).abs() / b

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")
