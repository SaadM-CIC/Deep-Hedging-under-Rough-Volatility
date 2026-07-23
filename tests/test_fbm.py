import numpy as np
import pytest
from dhrv.models.fbm import simulate_fbm_cholesky, estimate_hurst_exponent


def test_hurst_exponent_recovered_across_values():
    """L'exposant de Hurst estimé doit correspondre au H utilisé pour simuler,
    à tolérance ~0.02-0.05 (critère explicite de la roadmap, Phase 4)."""
    for H_true in [0.1, 0.3, 0.5, 0.7]:
        _, B = simulate_fbm_cholesky(H=H_true, T=1.0, n_steps=500, n_paths=2_000, seed=42)
        t_grid = np.linspace(0.0, 1.0, 501)
        H_hat = estimate_hurst_exponent(B, t_grid)
        assert abs(H_hat - H_true) < 0.05, f"H={H_true}: estimé {H_hat:.4f}"


def test_fbm_starts_at_zero():
    _, B = simulate_fbm_cholesky(H=0.3, T=1.0, n_steps=100, n_paths=10, seed=1)
    assert np.all(B[:, 0] == 0.0)


def test_fbm_variance_matches_covariance_formula():
    """Var(B^H_t) doit matcher t^{2H} (cas particulier t=s de la formule de covariance)."""
    H = 0.2
    _, B = simulate_fbm_cholesky(H=H, T=1.0, n_steps=200, n_paths=100_000, seed=7)
    t_grid = np.linspace(0.0, 1.0, 201)

    idx = 100
    empirical_var = B[:, idx].var()
    expected_var = t_grid[idx] ** (2 * H)
    assert empirical_var == pytest.approx(expected_var, rel=0.1)