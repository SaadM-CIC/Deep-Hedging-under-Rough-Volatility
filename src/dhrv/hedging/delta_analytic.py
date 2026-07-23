import numpy as np
from numpy.typing import NDArray

from dhrv.models.black_scholes import bs_call_price, bs_call_delta, simulate_gbm_paths


def bs_delta_hedge_pnl(
    S0: float,
    mu: float,
    sigma: float,
    K: float,
    r: float,
    T: float,
    n_rebal: int,
    n_paths: int,
    kappa: float = 0.0,
    seed: int | None = None,
) -> NDArray:
    """P&L final d'un portefeuille auto-financé (option courte + delta BS + cash),
    rebalancé n_rebal fois entre 0 et T.

    kappa : cout de transaction proportionnel.
    """
    t_grid, S = simulate_gbm_paths(S0, mu, sigma, T, n_rebal, n_paths, seed)
    dt = T / n_rebal

    V0 = bs_call_price(S0, K, r, sigma, T)
    delta_prev = bs_call_delta(S0, K, r, sigma, T)
    cash = np.full(n_paths, V0 - delta_prev * S0 - kappa * np.abs(delta_prev) * S0)

    for i in range(1, n_rebal + 1):
        cash = cash * np.exp(r * dt)
        tau = T - t_grid[i]
        if tau > 1e-12:
            delta_new = bs_call_delta(S[:, i], K, r, sigma, tau)
        else:
            delta_new = (S[:, i] > K).astype(float)
        trade = delta_new - delta_prev
        cash -= trade * S[:, i] + kappa * np.abs(trade) * S[:, i]
        delta_prev = delta_new

    payoff = np.maximum(S[:, -1] - K, 0.0)
    portfolio_value = delta_prev * S[:, -1] + cash
    return portfolio_value - payoff




