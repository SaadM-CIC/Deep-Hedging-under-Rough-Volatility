import torch


def entropic_risk(pnl: torch.Tensor, lam: float) -> torch.Tensor:
    """rho_lambda(X) = (1/lambda) * log E[exp(-lambda*X)], X = P&L terminal.

    Implementation stable via logsumexp (evite l'overflow de exp direct,
    cf. mise en garde Phase 7 sur lambda trop grand).
    """
    n = pnl.shape[0]
    return (torch.logsumexp(-lam * pnl, dim=0) - torch.log(torch.tensor(float(n)))) / lam

