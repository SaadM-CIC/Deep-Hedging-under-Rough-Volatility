import numpy as np
from numpy.typing import NDArray

from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.black_scholes import bs_call_delta


def rbergomi_proxy_hedge_pnl(
    model: RBergomiModel,
    S0: float,
    K: float,
    r: float,
    T: float,
    n_rebal: int,
    n_paths: int,
    kappa: float = 0.0,
    seed: int | None = None,
) -> NDArray:
    """P&L de couverture discrete sous rBergomi, delta = proxy BS avec vol
    instantanee sqrt(v_t). Ce n'est PAS un delta exact (rBergomi n'en a pas,
    processus non-markovien) -- c'est l'approximation documentee par la
    roadmap (Phase 6), son erreur residuelle est le resultat scientifique
    attendu de cette phase, pas un defaut a corriger.
    """
    t_grid, S, v = model.simulate_hybrid(S0=S0, T=T, n_steps=n_rebal, n_paths=n_paths, seed=seed)
    dt = T / n_rebal

    delta_prev = bs_call_delta(S[:, 0], K, r, np.sqrt(v[:, 0]), T)
    price0 = _bs_price_proxy(S0, K, r, np.sqrt(v[0, 0]), T)
    cash = np.full(n_paths, price0 - delta_prev[0] * S0 - kappa * np.abs(delta_prev[0]) * S0)
    # NB: price0 identique pour tous les paths (v[:,0] = xi0 constant a t=0)
    

    for i in range(1, n_rebal + 1):
        cash = cash * np.exp(r * dt)
        tau = T - t_grid[i]
        if tau > 1e-12:
            delta_new = bs_call_delta(S[:, i], K, r, np.sqrt(v[:, i]), tau)
        else:
            delta_new = (S[:, i] > K).astype(float)
        trade = delta_new - delta_prev
        cash -= trade * S[:, i] + kappa * np.abs(trade) * S[:, i]
        delta_prev = delta_new

    payoff = np.maximum(S[:, -1] - K, 0.0)
    portfolio_value = delta_prev * S[:, -1] + cash
    return portfolio_value - payoff


def _bs_price_proxy(S: float, K: float, r: float, sigma: float, T: float) -> float:
    from dhrv.models.black_scholes import bs_call_price
    return bs_call_price(S, K, r, sigma, T)