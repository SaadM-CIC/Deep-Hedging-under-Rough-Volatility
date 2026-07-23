import numpy as np
from numpy.typing import NDArray


def fbm_covariance_matrix(t_grid: NDArray, H: float) -> NDArray:
    """Matrice de covariance exacte du fBM : Cov(B^H_t, B^H_s) = 0.5*(t^2H + s^2H - |t-s|^2H)."""
    T = t_grid[:, None]
    S = t_grid[None, :]
    return 0.5 * (T ** (2 * H) + S ** (2 * H) - np.abs(T - S) ** (2 * H))


def simulate_fbm_cholesky(
    H: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Simulation exacte de fBM par décomposition de Cholesky de la matrice de covariance.

    Coûteux en O(n^2) mémoire / O(n^3) calcul (une seule factorisation, réutilisée pour
    tous les paths) — sert de référence lente mais correcte pour valider le Hybrid scheme
    (Phase 5).

    Returns
    -------
    t_grid : shape (n_steps + 1,)
    B : shape (n_paths, n_steps + 1), B[:, 0] = 0
    """
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # on exclut t=0 (covariance nulle, dégénère la factorisation) et le rajoute après
    t_interior = t_grid[1:]
    cov = fbm_covariance_matrix(t_interior, H)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(t_interior)))

    Z = rng.standard_normal(size=(n_paths, len(t_interior)))
    B_interior = Z @ L.T

    B = np.concatenate([np.zeros((n_paths, 1)), B_interior], axis=1)
    return t_grid, B


def estimate_hurst_exponent(B: NDArray, t_grid: NDArray, n_lags: int = 20) -> float:
    """Estime H par régression log-log de la variation quadratique moyenne en fonction du lag.

    E[(B_{t+lag} - B_t)^2] ~ lag^{2H}  =>  régresser log(var) sur log(lag), pente = 2H.
    """
    dt = t_grid[1] - t_grid[0]
    max_lag = min(n_lags, B.shape[1] // 4)
    lags = np.arange(1, max_lag + 1)

    log_lags = []
    log_vars = []
    for lag in lags:
        increments = B[:, lag:] - B[:, :-lag]
        var = np.mean(increments**2)
        log_lags.append(np.log(lag * dt))
        log_vars.append(np.log(var))

    slope, _ = np.polyfit(log_lags, log_vars, 1)
    return slope / 2.0