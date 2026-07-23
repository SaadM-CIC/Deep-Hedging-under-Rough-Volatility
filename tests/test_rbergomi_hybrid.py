import time
import numpy as np
import pytest
from scipy.stats import ks_2samp

from dhrv.models.rbergomi import RBergomiModel


def test_hybrid_matches_exact_moments():
    """Moments empiriques (moyenne, variance, skew de log S_T) doivent matcher
    entre Hybrid scheme et simulation exacte (Cholesky), a tolerance statistique."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    T, n_steps, n_paths = 1.0, 50, 30_000

    _, S_exact, _ = model.simulate_exact(S0=100.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=1)
    _, S_hybrid, _ = model.simulate_hybrid(S0=100.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=2)

    log_exact = np.log(S_exact[:, -1])
    log_hybrid = np.log(S_hybrid[:, -1])

    assert log_hybrid.mean() == pytest.approx(log_exact.mean(), abs=0.02)
    assert log_hybrid.var() == pytest.approx(log_exact.var(), rel=0.15)

    stat, p_value = ks_2samp(log_exact, log_hybrid)
    assert p_value > 0.01, f"KS test rejette H0 (p={p_value:.4f})"


def test_hybrid_much_faster_than_exact():
    """Le Hybrid scheme doit être nettement plus rapide que Cholesky sur une grille plus fine."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    T, n_steps, n_paths = 1.0, 200, 5_000

    t0 = time.perf_counter()
    model.simulate_exact(S0=100.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=1)
    time_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.simulate_hybrid(S0=100.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=2)
    time_hybrid = time.perf_counter() - t0

    print(f"exact: {time_exact:.3f}s, hybrid: {time_hybrid:.3f}s, speedup: {time_exact/time_hybrid:.1f}x")
    assert time_hybrid < time_exact