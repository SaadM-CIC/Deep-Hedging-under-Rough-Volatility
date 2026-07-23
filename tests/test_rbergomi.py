import numpy as np
import pytest

from dhrv.models.rbergomi import RBergomiModel


def test_volterra_variance_matches_closed_form():
    """Var(Z_hat_t) doit correspondre à la formule fermée t^(2H)/(2H)."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    T, n_steps, n_paths = 1.0, 50, 50_000
    t_grid, S, v = model.simulate_exact(S0=100.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=1)

    idx = n_steps // 2
    t = t_grid[idx]
    Z_hat = (np.log(v[:, idx] / model.xi0) + 0.5 * model.eta**2 * t ** (2 * model.H)) / (
        model.eta * np.sqrt(2 * model.H)
    )
    empirical_var = Z_hat.var()
    expected_var = t ** (2 * model.H) / (2 * model.H)
    assert empirical_var == pytest.approx(expected_var, rel=0.1)


def test_forward_variance_martingale_property():
    """E[v_t] doit rester proche de xi0 (propriété de martingale de la construction)."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    t_grid, S, v = model.simulate_exact(S0=100.0, T=1.0, n_steps=50, n_paths=50_000, seed=2)

    for idx in [10, 25, 50]:
        assert v[:, idx].mean() == pytest.approx(model.xi0, rel=0.1)


def test_price_paths_positive_and_finite():
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    t_grid, S, v = model.simulate_exact(S0=100.0, T=1.0, n_steps=50, n_paths=1_000, seed=3)

    assert np.all(S > 0)
    assert np.all(np.isfinite(S))
    assert np.allclose(S[:, 0], 100.0)