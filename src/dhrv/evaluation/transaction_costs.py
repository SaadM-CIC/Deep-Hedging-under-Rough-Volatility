import numpy as np
import torch

from dhrv.models.black_scholes import bs_call_price
from dhrv.models.black_scholes_torch import simulate_gbm_paths_torch
from dhrv.hedging.delta_analytic import bs_delta_hedge_pnl
from dhrv.hedging.train import train_bs_hedger
from dhrv.evaluation.metrics import summary_stats
from dhrv.evaluation.compare import _eval_ffn_rollout, TRAIN_SEED, TEST_SEED


def run_transaction_cost_sweep(
    kappas: list[float],
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 0.25,
    r: float = 0.0,
    sigma: float = 0.2,
    n_steps: int = 20,
    n_paths_train: int = 4_000,
    n_epochs: int = 300,
    n_paths_test: int = 20_000,
) -> dict:
    """Reentraine (deep) et reevalue (classique+deep) pour chaque kappa de la
    grille, sous BS. Le deep hedger est REENTRAINE a chaque kappa (piege
    explicite roadmap Phase 11 : jamais reutiliser un reseau entraine a
    kappa=0 sur un test kappa>0).
    """
    mu = r
    results = {}

    for kappa in kappas:
        pnl_analytic = bs_delta_hedge_pnl(
            S0, mu, sigma, K, r, T, n_steps, n_paths_test, kappa=kappa, seed=TEST_SEED
        )

        ffn = train_bs_hedger(
            S0, mu, sigma, K, r, T, n_steps, n_paths_train, n_epochs,
            kappa=kappa, seed=TRAIN_SEED,
        )
        ffn.eval()
        _, S_test = simulate_gbm_paths_torch(S0, mu, sigma, T, n_steps, n_paths_test)
        p0 = bs_call_price(S0, K, r, sigma, T)
        pnl_deep = _eval_ffn_rollout_with_cost(ffn, S_test, K, T, n_steps, n_paths_test, kappa) + p0

        results[kappa] = {
            "analytic_delta": summary_stats(pnl_analytic),
            "deep_hedging": summary_stats(pnl_deep),
        }
        print(f"kappa={kappa}: std_analytic={results[kappa]['analytic_delta']['std']:.4f}, "
              f"std_deep={results[kappa]['deep_hedging']['std']:.4f}")

    return results


def _eval_ffn_rollout_with_cost(model, S_seq, K, T, n_steps, n_paths, kappa):
    """Variante de _eval_ffn_rollout (compare.py) qui inclut le cout de
    transaction dans le P&L d'evaluation -- coherent avec ce que le reseau
    a ete entraine a minimiser a ce kappa."""
    t_grid = torch.linspace(0.0, T, n_steps + 1)
    delta_prev = torch.zeros(n_paths)
    cash_gain = torch.zeros(n_paths)
    cost_total = torch.zeros(n_paths)
    with torch.no_grad():
        for i in range(n_steps):
            tau = T - t_grid[i]
            log_moneyness = torch.log(S_seq[:, i] / K)
            tau_norm = (tau / T).expand(n_paths)
            x = torch.stack([log_moneyness, tau_norm, delta_prev], dim=-1)
            delta_new = model(x)
            cost_total = cost_total + kappa * torch.abs(delta_new - delta_prev) * S_seq[:, i]
            cash_gain = cash_gain + delta_new * (S_seq[:, i + 1] - S_seq[:, i])
            delta_prev = delta_new
    payoff = torch.clamp(S_seq[:, -1] - K, min=0.0)
    return (cash_gain - payoff - cost_total).numpy()


def performance_gap(results: dict) -> dict[float, float]:
    """Ecart de variance (analytic - deep), positif = deep hedging gagne."""
    return {
        kappa: r["analytic_delta"]["std"] ** 2 - r["deep_hedging"]["std"] ** 2
        for kappa, r in results.items()
    }


if __name__ == "__main__":
    kappas = [0.0, 0.0005, 0.001, 0.002, 0.005]
    results = run_transaction_cost_sweep(kappas)
    gaps = performance_gap(results)
    print("\nEcart de variance (analytic_var - deep_var), positif = deep hedging gagne:")
    for kappa, gap in gaps.items():
        print(f"  kappa={kappa}: gap={gap:.4f}")