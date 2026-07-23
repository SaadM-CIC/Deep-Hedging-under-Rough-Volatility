import numpy as np
import torch

from dhrv.models.heston import HestonModel
from dhrv.models.heston_torch import simulate_heston_paths_torch
from dhrv.hedging.delta_heston import heston_delta_hedge_pnl
from dhrv.hedging.train import train_heston_hedger
from dhrv.evaluation.metrics import summary_stats


def _eval_heston_deep(model, heston_true: HestonModel, S0, v0, mu, K, T, n_steps, n_paths, seed):
    """Evalue un deep hedger (entraine sous des parametres de reference) sur
    des trajectoires generees avec heston_true (potentiellement perturbe),
    SANS reentrainement -- c'est le coeur du test de robustesse."""
    model.eval()
    with torch.no_grad():
        t_grid, S, v = simulate_heston_paths_torch(heston_true, S0, v0, mu, T, n_steps, n_paths, seed=seed)
        delta_prev = torch.zeros(n_paths)
        cash_gain = torch.zeros(n_paths)
        for i in range(n_steps):
            tau = T - t_grid[i]
            S_t, v_t = S[:, i], v[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = (tau / T).expand(n_paths)
            x = torch.stack([log_moneyness, tau_norm, v_t, delta_prev], dim=-1)
            delta_new = model(x)
            cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
            delta_prev = delta_new
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        p0 = heston_true.call_price(np.array([S0]), np.array([v0]), K, T)[0]
        return (cash_gain - payoff).numpy() + p0


def run_robustness_sweep(
    rho_shocks: list[float],
    S0: float = 100.0,
    v0: float = 0.04,
    K: float = 100.0,
    T: float = 0.25,
    r: float = 0.0,
    n_steps: int = 20,
    n_paths_train: int = 4_000,
    n_epochs: int = 300,
    n_paths_test: int = 20_000,
) -> dict:
    """Entraine deep hedger + calibre le hedge classique sous un Heston de
    reference (rho=-0.7), puis EVALUE les deux (sans reentrainement) sous
    des Heston perturbes ou seul rho differe (mauvaise specification du
    levier). Le hedge classique utilise le MEME modele de reference (mal
    specifie) que le deep hedger -- comparaison equitable de mauvaise
    specification, pas d'un oracle vs un modele.
    """
    mu = r
    kappa_h, theta_h, xi_h = 2.0, 0.04, 0.3
    rho_ref = -0.7

    heston_ref = HestonModel(kappa=kappa_h, theta=theta_h, xi=xi_h, rho=rho_ref, r=r)
    deep_model = train_heston_hedger(
        heston_ref, S0, v0, mu, K, T, n_steps, n_paths_train, n_epochs, seed=42
    )

    results = {}
    for rho_shocked in rho_shocks:
        heston_true = HestonModel(kappa=kappa_h, theta=theta_h, xi=xi_h, rho=rho_shocked, r=r)

        # classique : delta MV calcule SOUS LE MODELE DE REFERENCE (mal specifie),
        # applique aux vraies trajectoires perturbees -- coherent avec le deep hedger
        # qui lui aussi n'a jamais vu rho_shocked.
        pnl_classic = _classic_hedge_misspecified(
            heston_ref, heston_true, S0, v0, mu, K, T, n_steps, n_paths_test, seed=999
        )

        pnl_deep = _eval_heston_deep(
            deep_model, heston_true, S0, v0, mu, K, T, n_steps, n_paths_test, seed=999
        )

        results[rho_shocked] = {
            "classic_mv_delta": summary_stats(pnl_classic),
            "deep_hedging": summary_stats(pnl_deep),
        }
        print(f"rho_shocked={rho_shocked}: std_classic={results[rho_shocked]['classic_mv_delta']['std']:.4f}, "
              f"std_deep={results[rho_shocked]['deep_hedging']['std']:.4f}")

    return results


def _classic_hedge_misspecified(heston_ref, heston_true, S0, v0, mu, K, T, n_steps, n_paths, seed):
    """Simule sous heston_true, mais calcule le delta MV avec les parametres
    de heston_ref (mauvaise specification), pour une comparaison equitable
    avec le deep hedger (lui aussi entraine sous heston_ref uniquement)."""
    t_grid, S, v = heston_true.simulate(S0, v0, mu, T, n_steps, n_paths, seed=seed)
    dt = T / n_steps
    r = heston_ref.r

    V0 = heston_ref.call_price(np.array([S0]), np.array([v0]), K, T)[0]
    delta_prev = heston_ref.call_delta_min_variance(np.array([S0]), np.array([v0]), K, T)
    cash = np.full(n_paths, V0 - delta_prev[0] * S0)

    for i in range(1, n_steps + 1):
        cash = cash * np.exp(r * dt)
        tau = T - t_grid[i]
        if tau > 1e-12:
            delta_new = heston_ref.call_delta_min_variance(S[:, i], v[:, i], K, tau)
        else:
            delta_new = (S[:, i] > K).astype(float)
        cash -= (delta_new - delta_prev) * S[:, i]
        delta_prev = delta_new

    payoff = np.maximum(S[:, -1] - K, 0.0)
    portfolio_value = delta_prev * S[:, -1] + cash
    return portfolio_value - payoff


if __name__ == "__main__":
    rho_shocks = [-0.7, -0.5, -0.3, 0.0]  # -0.7 = pas de choc (reference), le reste = mauvaise specification croissante
    results = run_robustness_sweep(rho_shocks)

    print("\nDegradation relative (std_shocked / std_reference):")
    ref_classic = results[-0.7]["classic_mv_delta"]["std"]
    ref_deep = results[-0.7]["deep_hedging"]["std"]
    for rho, r in results.items():
        print(f"  rho={rho}: classic x{r['classic_mv_delta']['std']/ref_classic:.2f}, "
              f"deep x{r['deep_hedging']['std']/ref_deep:.2f}")