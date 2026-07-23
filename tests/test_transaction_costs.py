from dhrv.hedging.delta_analytic import bs_delta_hedge_pnl


def test_transaction_costs_reduce_pnl_mean():
    """Sanity check : des couts positifs doivent degrader (baisser) le P&L moyen
    par rapport a kappa=0, toutes choses egales par ailleurs."""
    S0, mu, sigma, K, r, T = 100.0, 0.0, 0.2, 100.0, 0.0, 0.25
    n_rebal, n_paths, seed = 20, 20_000, 42

    pnl_no_cost = bs_delta_hedge_pnl(S0, mu, sigma, K, r, T, n_rebal, n_paths, kappa=0.0, seed=seed)
    pnl_with_cost = bs_delta_hedge_pnl(S0, mu, sigma, K, r, T, n_rebal, n_paths, kappa=0.001, seed=seed)

    assert pnl_with_cost.mean() < pnl_no_cost.mean()