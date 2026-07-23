import numpy as np

from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.implied_vol import implied_vol


def _atm_skew(model: RBergomiModel, S0: float, T: float, n_paths: int, seed: int) -> float:
    """Skew ATM : d(implied vol)/d(log-moneyness) estime par difference finie centree
    autour de K=S0, sur des trajectoires simulees par Hybrid scheme."""
    n_steps = max(int(T * 50), 10)
    _, S, _ = model.simulate_hybrid(S0=S0, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    S_T = S[:, -1]
    r = 0.0

    eps = 0.02
    K_down = S0 * np.exp(-eps)
    K_up = S0 * np.exp(eps)

    price_down = np.exp(-r * T) * np.maximum(S_T - K_down, 0.0).mean()
    price_up = np.exp(-r * T) * np.maximum(S_T - K_up, 0.0).mean()

    iv_down = implied_vol(price_down, S0, K_down, r, T)
    iv_up = implied_vol(price_up, S0, K_up, r, T)

    return (iv_up - iv_down) / (2 * eps)


def test_atm_skew_explodes_as_short_maturity_power_law():
    """|skew(T)| doit croitre quand T diminue, en loi de puissance d'exposant ~ H-0.5
    (skew ~ T^(H-0.5)), signature du rBergomi absente de Heston (roadmap Phase 5)."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    S0, n_paths = 100.0, 200_000

    T1, T2 = 0.02, 0.08
    skew1 = abs(_atm_skew(model, S0, T1, n_paths, seed=1))
    skew2 = abs(_atm_skew(model, S0, T2, n_paths, seed=2))

    # skew plus fort a maturite plus courte
    assert skew1 > skew2

    # exposant empirique ~ H - 0.5, tolerance large (bruit MC + eps fini)
    empirical_exponent = np.log(skew1 / skew2) / np.log(T1 / T2)
    expected_exponent = model.H - 0.5
    assert empirical_exponent == pytest_approx_range(expected_exponent, tol=0.25)


def pytest_approx_range(expected, tol):
    class _Range:
        def __eq__(self, other):
            return abs(other - expected) < tol
    return _Range()