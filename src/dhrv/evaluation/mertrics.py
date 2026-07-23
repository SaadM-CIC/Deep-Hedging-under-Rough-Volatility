import numpy as np
from numpy.typing import NDArray


def summary_stats(pnl: NDArray, alpha: float = 0.95) -> dict:
    """Statistiques standard de P&L : moyenne, std, VaR, CVaR (sur la perte = -pnl).

    Unique source de verite pour ces stats -- reutilisee par compare.py,
    transaction_costs.py, robustness.py (Phases 11-12), jamais recalculee
    ailleurs (roadmap, modules reutilisables).
    """
    loss = -pnl
    var = float(np.quantile(loss, alpha))
    cvar = float(loss[loss >= var].mean())
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        f"VaR{int(alpha * 100)}": var,
        f"CVaR{int(alpha * 100)}": cvar,
    }