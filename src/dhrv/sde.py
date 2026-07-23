import numpy as np
from numpy.typing import NDArray


def simulate_brownian_motion(
    n_paths: int,
    n_steps: int,
    T: float,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Simule n_paths trajectoires de mouvement brownien standard sur [0, T].

    Returns
    -------
    t_grid : shape (n_steps + 1,)
    W : shape (n_paths, n_steps + 1), W[:, 0] = 0
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    t_grid = np.linspace(0.0, T, n_steps + 1)
    return t_grid, W


def euler_maruyama(
    drift,
    diffusion,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    dW: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Schéma d'Euler-Maruyama générique pour dX_t = drift(t, X_t) dt + diffusion(t, X_t) dW_t.

    drift, diffusion : callables (t: float, x: NDArray[n_paths]) -> NDArray[n_paths]
    dW : increments browniens optionnels, shape (n_paths, n_steps). Si fournis,
         permet d'imposer les mêmes tirages qu'une solution de référence
         (common random numbers) — utilisé pour les tests de convergence forte
         et réutilisé en Phase 6 (delta pathwise).

    Returns
    -------
    t_grid : shape (n_steps + 1,)
    X : shape (n_paths, n_steps + 1), X[:, 0] = x0
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    if dW is None:
        rng = np.random.default_rng(seed)
        dW = rng.normal(loc=0.0, scale=sqrt_dt, size=(n_paths, n_steps))
    else:
        if dW.shape != (n_paths, n_steps):
            raise ValueError(f"dW shape {dW.shape} != expected {(n_paths, n_steps)}")

    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    t_grid = np.linspace(0.0, T, n_steps + 1)

    for i in range(n_steps):
        t = t_grid[i]
        x = X[:, i]
        X[:, i + 1] = x + drift(t, x) * dt + diffusion(t, x) * dW[:, i]

    return t_grid, X