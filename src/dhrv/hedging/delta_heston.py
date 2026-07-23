import numpy as np
from numpy.typing import NDArray

from dhrv.models.heston import HestonModel
from dhrv.models.black_scholes import bs_call_delta


def heston_delta_hedge_pnl(
    model: HestonModel,
    S0: float,
    v0: float,
    mu: float,
    K: float,
    T: float,
    n_rebal: int,
    n_paths: int,
    delta_type: str,
    kappa: float = 0.0,
    seed: int | None = None,
) -> NDArray:
    """P&L de couverture discrète sous Heston.

    delta_type : "heston" (delta correct, P1) ou "bs_naive"
                 (delta BS avec vol instantanée sqrt(v_t)).
    """
    t_grid, S, v = model.simulate(S0, v0, mu, T, n_rebal, n_paths, seed=seed)
    dt = T / n_rebal
    r = model.r

    V0 = model.call_price(np.array([S0]), np.array([v0]), K, T)[0]
    delta_prev = _compute_delta(model, S[:, 0], v[:, 0], K, T, delta_type)
    cash = np.full(n_paths, V0 - delta_prev * S0 - kappa * np.abs(delta_prev) * S0)

    for i in range(1, n_rebal + 1):
        cash = cash * np.exp(r * dt)
        tau = T - t_grid[i]
        delta_new = _compute_delta(model, S[:, i], v[:, i], K, tau, delta_type)
        trade = delta_new - delta_prev
        cash -= trade * S[:, i] + kappa * np.abs(trade) * S[:, i]
        delta_prev = delta_new

    payoff = np.maximum(S[:, -1] - K, 0.0)
    portfolio_value = delta_prev * S[:, -1] + cash
    return portfolio_value - payoff


def _compute_delta(
    model: HestonModel, S: NDArray, v: NDArray, K: float, tau: float, delta_type: str
) -> NDArray:
    if delta_type == "heston":
        return model.call_delta(S, v, K, tau)
    elif delta_type == "heston_mv":
        return model.call_delta_min_variance(S, v, K, tau)
   
    elif delta_type == "bs_naive":
        if tau <= 1e-12:
            return (S > K).astype(float)
        return bs_call_delta(S, K, model.r, np.sqrt(v), tau)
    else:
        raise ValueError(f"delta_type inconnu : {delta_type}")