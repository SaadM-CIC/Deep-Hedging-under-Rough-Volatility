from dhrv.evaluation.robustness import run_robustness_sweep


def test_robustness_pnl_finite_across_shocks():
    """Sanity check : le P&L doit rester fini pour toute perturbation de rho."""
    results = run_robustness_sweep(rho_shocks=[-0.7, 0.0], n_epochs=150, n_paths_test=5_000)
    for rho, stats in results.items():
        assert stats["classic_mv_delta"]["std"] > 0
        assert stats["deep_hedging"]["std"] > 0