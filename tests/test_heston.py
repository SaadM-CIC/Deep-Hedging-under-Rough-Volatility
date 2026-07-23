import numpy as np


from dhrv.models.heston import HestonModel
from dhrv.hedging.delta_heston import heston_delta_hedge_pnl


def test_fft_price_matches_mc_price():
    """Prix semi-analytique Heston doit matcher un prix Monte Carlo indépendant."""
    model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.02)
    S0, v0, K, T = 100.0, 0.04, 100.0, 1.0
    n_paths = 300_000

    price_analytic = model.call_price(np.array([S0]), np.array([v0]), K, T)[0]

    _, S, _ = model.simulate(S0, v0, mu=model.r, T=T, n_steps=100, n_paths=n_paths, seed=7)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    price_mc = np.exp(-model.r * T) * payoff.mean()
    se = np.exp(-model.r * T) * payoff.std() / np.sqrt(n_paths)

    assert abs(price_analytic - price_mc) < 4 * se


def test_heston_mv_delta_beats_bs_naive_delta():
    """Le delta à variance minimale (corrigé du levier rho) doit battre le delta BS naif.
    Le P1 brut, lui, ne le bat pas nécessairement en présence de fort effet de levier —
    résultat documenté (Alexander & Nogueira 2007), pas un bug."""
    model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.02)
    S0, v0, mu, K, T = 100.0, 0.04, 0.05, 100.0, 1.0
    n_paths = 20_000
    n_rebal = 50

    pnl_mv = heston_delta_hedge_pnl(
        model, S0, v0, mu, K, T, n_rebal, n_paths, delta_type="heston_mv", seed=42
    )
    pnl_naive = heston_delta_hedge_pnl(
        model, S0, v0, mu, K, T, n_rebal, n_paths, delta_type="bs_naive", seed=42
    )

    assert pnl_mv.var() < pnl_naive.var()






