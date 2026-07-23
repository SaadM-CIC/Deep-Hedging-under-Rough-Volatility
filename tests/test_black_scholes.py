import numpy as np

from dhrv.models.black_scholes import bs_call_price, simulate_gbm_paths
from dhrv.hedging.delta_analytic import bs_delta_hedge_pnl


def test_mc_price_matches_closed_form():
    """Prix Monte Carlo sous mesure risque-neutre (mu=r) doit matcher le prix fermé."""
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.2, 1.0
    n_paths = 500_000

    price_cf = bs_call_price(S0, K, r, sigma, T)

    _, S = simulate_gbm_paths(S0, mu=r, sigma=sigma, T=T, n_steps=1, n_paths=n_paths, seed=1)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    price_mc = np.exp(-r * T) * payoff.mean()
    se = np.exp(-r * T) * payoff.std() / np.sqrt(n_paths)

    assert abs(price_mc - price_cf) < 3 * se


def test_hedging_pnl_variance_decreases_with_frequency():
    """Var(P&L) doit décroître (globalement) quand la fréquence de rebalancement augmente."""
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.2, 1.0
    mu = 0.05  # drift physique, distinct de r
    n_paths = 20_000

    freqs = [1, 4, 12, 52, 252]
    variances = []
    for n in freqs:
        pnl = bs_delta_hedge_pnl(S0, mu, sigma, K, r, T, n_rebal=n, n_paths=n_paths, seed=42)
        variances.append(pnl.var())

    assert variances[-1] < variances[0]
    # tendance décroissante globale, tolérance pour le bruit Monte Carlo
    assert all(variances[i + 1] <= variances[i] * 1.15 for i in range(len(variances) - 1))