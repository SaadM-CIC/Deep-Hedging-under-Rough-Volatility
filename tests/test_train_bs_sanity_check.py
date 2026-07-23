import torch

from dhrv.hedging.train import train_bs_hedger
from dhrv.models.black_scholes import bs_call_delta
from dhrv.models.black_scholes_torch import simulate_gbm_paths_torch


def test_deep_hedger_bs_matches_analytic_delta():
    """SANITY CHECK BLOQUANT (roadmap Phase 9) : le deep hedger entraine sous BS
    sans couts doit converger vers une politique proche du delta BS analytique.
    Si ce test echoue, le pipeline d'entrainement est casse -- ne pas aller
    plus loin (rBergomi, Heston) tant qu'il ne passe pas.
    IMPORTANT : entrainement fait sous mu=r (mesure risque-neutre). Sous mu != r,
    l'entropic risk minimizer capte legitimement la prime de risque (mu-r) et
    devie du delta hedge pur -- ce n'est pas un defaut du pipeline, mais ca
    confondrait le sanity check. Isoler la replication pure exige mu=r ici.
    Si ce test echoue, le pipeline d'entrainement est casse -- ne pas aller
    plus loin (rBergomi, Heston) tant qu'il ne passe pas.
    
    """
    S0, r, sigma, K, T = 100.0, 0.0, 0.2, 100.0, 0.25
    mu = r  # mesure risque-neutre pour isoler la replication pure
    n_steps, n_paths, n_epochs = 20, 4_000, 300

    model = train_bs_hedger(
        S0, mu, sigma, K, r, T, n_steps, n_paths, n_epochs, lam=1.0, lr=1e-3, seed=42
    )

    model.eval()
    with torch.no_grad():
        t_grid, S = simulate_gbm_paths_torch(S0, mu, sigma, T, n_steps, n_paths=10_000)

        delta_prev = torch.zeros(10_000)
        max_abs_diff = 0.0
        for i in range(n_steps):
            tau = (T - t_grid[i]).item()
            if tau < 1e-6:
                break
            S_t = S[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = torch.full_like(S_t, tau / T)
            x = torch.stack([log_moneyness, tau_norm, delta_prev], dim=-1)
            delta_new = model(x)

            delta_analytic = bs_call_delta(S_t.numpy(), K, r, sigma, tau)
            diff = (delta_new.numpy() - delta_analytic)
            max_abs_diff = max(max_abs_diff, abs(diff).mean())

            delta_prev = delta_new

        print(f"ecart moyen absolu deep-hedger vs delta BS analytique: {max_abs_diff:.4f}")
        assert max_abs_diff < 0.15