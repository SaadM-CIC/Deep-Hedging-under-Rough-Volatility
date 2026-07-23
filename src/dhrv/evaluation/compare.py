import numpy as np
import torch

from dhrv.models.black_scholes import bs_call_price
from dhrv.models.heston import HestonModel
from dhrv.models.rbergomi import RBergomiModel
from dhrv.hedging.delta_analytic import bs_delta_hedge_pnl
from dhrv.hedging.delta_heston import heston_delta_hedge_pnl
from dhrv.hedging.delta_rbergomi_proxy import rbergomi_proxy_hedge_pnl
from dhrv.hedging.train import (
    train_bs_hedger, train_heston_hedger, train_rbergomi_hedger,
)
from dhrv.models.black_scholes_torch import simulate_gbm_paths_torch
from dhrv.models.heston_torch import simulate_heston_paths_torch
from dhrv.models.rbergomi_torch import simulate_rbergomi_paths_torch
from dhrv.evaluation.metrics import summary_stats


TRAIN_SEED = 42       # seeds d'entrainement des deep hedgers
TEST_SEED = 999_000    # seeds disjoints -- jeu de test jamais vu a l'entrainement


def _eval_ffn_rollout(model, S_seq, extra_seq, K, T, n_steps, n_paths):
    """Rollout d'evaluation generique pour un FFNHedger deja entraine.
    extra_seq : liste de tenseurs (n_paths, n_steps) de features additionnelles
    (ex. v_t), vide pour BS.
    """
    t_grid = torch.linspace(0.0, T, n_steps + 1)
    delta_prev = torch.zeros(n_paths)
    cash_gain = torch.zeros(n_paths)
    with torch.no_grad():
        for i in range(n_steps):
            tau = T - t_grid[i]
            log_moneyness = torch.log(S_seq[:, i] / K)
            tau_norm = (tau / T).expand(n_paths)
            feats = [log_moneyness, tau_norm] + [e[:, i] for e in extra_seq] + [delta_prev]
            x = torch.stack(feats, dim=-1)
            delta_new = model(x)
            cash_gain = cash_gain + delta_new * (S_seq[:, i + 1] - S_seq[:, i])
            delta_prev = delta_new
    payoff = torch.clamp(S_seq[:, -1] - K, min=0.0)
    return (cash_gain - payoff).numpy()


def run_comparison(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 0.25,
    r: float = 0.0,
    n_steps: int = 20,
    n_paths_train: int = 4_000,
    n_epochs: int = 300,
    n_paths_test: int = 20_000,
) -> dict:
    """Entraine les 3 deep hedgers (seeds d'entrainement fixes), evalue les 4
    strategies par modele sous-jacent sur un jeu de test disjoint. Retourne
    un dict {modele: {methode: stats}}.
    """
    results: dict[str, dict[str, dict]] = {}

    # --- Black-Scholes ---
    sigma = 0.2
    mu = r  # mesure risque-neutre, coherent avec le sanity check Phase 9.1
    pnl_bs_analytic = bs_delta_hedge_pnl(S0, mu, sigma, K, r, T, n_steps, n_paths_test, seed=TEST_SEED)

    ffn_bs = train_bs_hedger(S0, mu, sigma, K, r, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED)
    ffn_bs.eval()
    _, S_test_bs = simulate_gbm_paths_torch(S0, mu, sigma, T, n_steps, n_paths_test)
    p0_bs = bs_call_price(S0, K, r, sigma, T)
    pnl_bs_deep = _eval_ffn_rollout(ffn_bs, S_test_bs, [], K, T, n_steps, n_paths_test) + p0_bs

    results["black_scholes"] = {
        "analytic_delta": summary_stats(pnl_bs_analytic),
        "deep_hedging": summary_stats(pnl_bs_deep),
    }

    # --- Heston ---
    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=r)
    v0 = 0.04
    pnl_heston_mv = heston_delta_hedge_pnl(
        heston, S0, v0, mu, K, T, n_steps, n_paths_test, delta_type="heston_mv", seed=TEST_SEED
    )

    ffn_heston = train_heston_hedger(
        heston, S0, v0, mu, K, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED
    )
    ffn_heston.eval()
    _, S_test_h, v_test_h = simulate_heston_paths_torch(
        heston, S0, v0, mu, T, n_steps, n_paths_test, seed=TEST_SEED
    )
    p0_heston = heston.call_price(np.array([S0]), np.array([v0]), K, T)[0]
    pnl_heston_deep = _eval_ffn_rollout(ffn_heston, S_test_h, [v_test_h], K, T, n_steps, n_paths_test) + p0_heston

    results["heston"] = {
        "analytic_delta_mv": summary_stats(pnl_heston_mv),
        "deep_hedging": summary_stats(pnl_heston_deep),
    }

    # --- rBergomi ---
    rbergomi = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    pnl_rbergomi_proxy = rbergomi_proxy_hedge_pnl(
        rbergomi, S0, K, r, T, n_steps, n_paths_test, seed=TEST_SEED
    )

    ffn_rbergomi = train_rbergomi_hedger(
        rbergomi, S0, K, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED
    )
    ffn_rbergomi.eval()
    _, S_test_r, v_test_r = simulate_rbergomi_paths_torch(
        rbergomi, S0, T, n_steps, n_paths_test, seed=TEST_SEED
    )
    p0_rbergomi = bs_call_price(S0, K, r, np.sqrt(rbergomi.xi0), T)  # meme convention que le proxy (Phase 6)
    pnl_rbergomi_deep = _eval_ffn_rollout(ffn_rbergomi, S_test_r, [v_test_r], K, T, n_steps, n_paths_test) + p0_rbergomi

    results["rbergomi"] = {
        "proxy_bs_delta": summary_stats(pnl_rbergomi_proxy),
        "deep_hedging": summary_stats(pnl_rbergomi_deep),
    }

    return results


def print_comparison_table(results: dict) -> None:
    for model_name, methods in results.items():
        print(f"\n=== {model_name} ===")
        for method_name, stats in methods.items():
            stats_str = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
            print(f"  {method_name:20s} : {stats_str}")


if __name__ == "__main__":
    results = run_comparison()
    print_comparison_table(results)