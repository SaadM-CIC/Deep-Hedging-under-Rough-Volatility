import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def bs_call_price(S: float | NDArray, K: float, r: float, sigma: float, T: float) -> float | NDArray:
    """Prix Black-Scholes fermé d'un call européen."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float | NDArray, K: float, r: float, sigma: float, tau: float | NDArray) -> float | NDArray:
    """Delta Black-Scholes fermé, tau = temps restant avant maturité."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Simule des trajectoires de mouvement brownien géométrique (solution exacte, pas d'Euler).

    dS_t = mu*S_t dt + sigma*S_t dW_t

    Returns
    -------
    t_grid : shape (n_steps + 1,)
    S : shape (n_paths, n_steps + 1), S[:, 0] = S0
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    t_grid = np.linspace(0.0, T, n_steps + 1)

    log_S = np.log(S0) + (mu - 0.5 * sigma**2) * t_grid[1:] + sigma * W
    log_S = np.concatenate([np.full((n_paths, 1), np.log(S0)), log_S], axis=1)
    return t_grid, np.exp(log_S)