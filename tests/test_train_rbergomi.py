import numpy as np
import torch

from dhrv.hedging.train import train_rbergomi_hedger
from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.rbergomi_torch import simulate_rbergomi_paths_torch
from dhrv.hedging.delta_rbergomi_proxy import rbergomi_proxy_hedge_pnl


def test_deep_hedger_rbergomi_variance_stable_across_seeds():
    """Critere Phase 9 : variance inter-seeds documentee (pas un seul run
    'chanceux'). On entraine 2 seeds independants et on rapporte la variance
    de l'ecart-type du P&L obtenu, en comparaison au proxy BS (Phase 6).
    """
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    S0, K, T = 100.0, 100.0, 0.25
    n_steps, n_paths, n_epochs = 20, 4_000, 300
    n_eval = 20_000

    stds = []
    for seed in [1, 2]:
        deep_model = train_rbergomi_hedger(
            model, S0, K, T, n_steps, n_paths, n_epochs, lam=1.0, lr=1e-3, seed=seed
        )
        deep_model.eval()
        with torch.no_grad():
            t_grid, S, v = simulate_rbergomi_paths_torch(
                model, S0, T, n_steps, n_paths=n_eval, seed=1000 + seed
            )
            delta_prev = torch.zeros(n_eval)
            cash_gain = torch.zeros(n_eval)
            for i in range(n_steps):
                tau = T - t_grid[i]
                S_t, v_t = S[:, i], v[:, i]
                log_moneyness = torch.log(S_t / K)
                tau_norm = (tau / T).expand(n_eval)
                x = torch.stack([log_moneyness, tau_norm, v_t, delta_prev], dim=-1)
                delta_new = deep_model(x)
                cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
                delta_prev = delta_new
            payoff = torch.clamp(S[:, -1] - K, min=0.0)
            pnl_deep = (cash_gain - payoff).numpy()
        stds.append(pnl_deep.std())
        print(f"seed={seed}: std deep hedger rBergomi = {pnl_deep.std():.4f}")

    pnl_proxy = rbergomi_proxy_hedge_pnl(model, S0, K, r=0.0, T=T, n_rebal=n_steps, n_paths=n_eval, seed=2000)
    print(f"std proxy BS = {pnl_proxy.std():.4f}")
    print(f"variance inter-seeds (std des std) = {np.std(stds):.4f}")

    # sanity check large : le deep hedger ne doit pas exploser par rapport au proxy
    assert all(s < 3 * pnl_proxy.std() for s in stds)