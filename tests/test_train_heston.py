import torch

from dhrv.hedging.train import train_heston_hedger
from dhrv.models.heston import HestonModel
from dhrv.models.heston_torch import simulate_heston_paths_torch
from dhrv.hedging.delta_heston import heston_delta_hedge_pnl


def test_deep_hedger_heston_variance_in_range_of_mv_delta():
    """Contrairement au sanity check BS (Phase 9.1, egalite de politique),
    il n'existe pas de politique fermee unique a retrouver exactement sous
    Heston avec ce reseau simple. Critere ici : la variance du P&L du deep
    hedger doit rester du meme ordre de grandeur (ou meilleure) que celle
    du delta a variance minimale (Phase 3), sur une evaluation independante.
    """
    kappa, theta, xi, rho, r = 2.0, 0.04, 0.3, -0.7, 0.0
    heston = HestonModel(kappa=kappa, theta=theta, xi=xi, rho=rho, r=r)
    S0, v0, K, T = 100.0, 0.04, 100.0, 0.25
    n_steps, n_paths, n_epochs = 20, 4_000, 300
    mu = r  # mesure risque-neutre, coherent avec le sanity check 9.1

    model = train_heston_hedger(
        heston, S0, v0, mu, K, T, n_steps, n_paths, n_epochs, lam=1.0, lr=1e-3, seed=42
    )

    model.eval()
    with torch.no_grad():
        n_eval = 20_000
        t_grid, S, v = simulate_heston_paths_torch(
            heston, S0, v0, mu, T, n_steps, n_paths=n_eval, seed=999
        )
        delta_prev = torch.zeros(n_eval)
        cash_gain = torch.zeros(n_eval)
        for i in range(n_steps):
            tau = T - t_grid[i]
            S_t, v_t = S[:, i], v[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = (tau / T).expand(n_eval)
            x = torch.stack([log_moneyness, tau_norm, v_t, delta_prev], dim=-1)
            delta_new = model(x)
            cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
            delta_prev = delta_new
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl_deep = (cash_gain - payoff).numpy()

    pnl_mv = heston_delta_hedge_pnl(
        heston, S0, v0, mu, K, T, n_rebal=n_steps, n_paths=n_eval, delta_type="heston_mv", seed=999
    )

    print(f"std deep hedger = {pnl_deep.std():.4f}, std heston_mv = {pnl_mv.std():.4f}")
    assert pnl_deep.std() < 3 * pnl_mv.std()