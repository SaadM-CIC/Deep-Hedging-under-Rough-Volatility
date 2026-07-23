import numpy as np
import pytest

from dhrv.sde import simulate_brownian_motion, euler_maruyama


def test_brownian_motion_variance_converges_to_t():
    """Var(B_t) doit converger vers t, pour plusieurs t et grand échantillon."""
    _, W = simulate_brownian_motion(n_paths=200_000, n_steps=252, T=1.0, seed=42)
    t_grid = np.linspace(0.0, 1.0, 253)

    for idx in [50, 126, 252]:
        empirical_var = W[:, idx].var()
        expected_var = t_grid[idx]
        assert empirical_var == pytest.approx(expected_var, rel=0.05)


def test_brownian_motion_starts_at_zero():
    _, W = simulate_brownian_motion(n_paths=10, n_steps=100, T=1.0, seed=1)
    assert np.all(W[:, 0] == 0.0)


def test_euler_maruyama_gbm_strong_convergence():
    """Erreur forte pathwise (common random numbers) doit décroître ~O(sqrt(dt))."""
    mu, sigma, x0, T = 0.05, 0.2, 100.0, 1.0
    n_paths = 20_000
    rng = np.random.default_rng(123)

    def drift(t, x):
        return mu * x

    def diffusion(t, x):
        return sigma * x

    steps_list = [50, 200, 800]
    errors = []

    for n_steps in steps_list:
        dt = T / n_steps
        dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        W_T = dW.sum(axis=1)
        X_exact = x0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)

        _, X_euler = euler_maruyama(
            drift, diffusion, x0, T, n_steps, n_paths, dW=dW
        )
        error = np.abs(X_euler[:, -1] - X_exact).mean()
        errors.append(error)

    # Erreur forte doit décroître quand n_steps augmente (O(sqrt(dt)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]