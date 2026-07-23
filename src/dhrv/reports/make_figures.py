import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dhrv.models.black_scholes import simulate_gbm_paths
from dhrv.models.heston import HestonModel
from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.implied_vol import implied_vol
from dhrv.models.black_scholes import bs_call_price
from dhrv.evaluation.compare import run_comparison
from dhrv.evaluation.transaction_costs import run_transaction_cost_sweep, performance_gap
from dhrv.evaluation.robustness import run_robustness_sweep
from dhrv.hedging.delta_analytic import bs_delta_hedge_pnl
from dhrv.hedging.delta_heston import heston_delta_hedge_pnl
from dhrv.hedging.delta_rbergomi_proxy import rbergomi_proxy_hedge_pnl
from dhrv.hedging.train import train_bs_hedger, train_heston_hedger, train_rbergomi_hedger
from dhrv.evaluation.compare import _eval_ffn_rollout, TRAIN_SEED, TEST_SEED
from dhrv.models.black_scholes_torch import simulate_gbm_paths_torch
from dhrv.models.heston_torch import simulate_heston_paths_torch
from dhrv.models.rbergomi_torch import simulate_rbergomi_paths_torch
FIGDIR = "reports/figures"
os.makedirs(FIGDIR, exist_ok=True)

STYLE = {"bs": "#185FA5", "heston": "#0F6E56", "rbergomi": "#A32D2D"}


def fig_trajectories_side_by_side(seed: int = 42) -> None:
    """Trajectoires simulees BS / Heston / rBergomi cote a cote (roadmap Phase 13)."""
    S0, T, n_steps, n_paths_plot = 100.0, 1.0, 252, 8

    _, S_bs = simulate_gbm_paths(S0, mu=0.05, sigma=0.2, T=T, n_steps=n_steps, n_paths=n_paths_plot, seed=seed)

    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.0)
    _, S_heston, _ = heston.simulate(S0, v0=0.04, mu=0.0, T=T, n_steps=n_steps, n_paths=n_paths_plot, seed=seed)

    rbergomi = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    _, S_rbergomi, _ = rbergomi.simulate_hybrid(S0=S0, T=T, n_steps=n_steps, n_paths=n_paths_plot, seed=seed)

    t = np.linspace(0, T, n_steps + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, S, title, color in zip(
        axes, [S_bs, S_heston, S_rbergomi],
        ["Black-Scholes", "Heston", "rBergomi (H=0.1)"],
        [STYLE["bs"], STYLE["heston"], STYLE["rbergomi"]],
    ):
        for i in range(n_paths_plot):
            ax.plot(t, S[i], color=color, alpha=0.7, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("t")
    axes[0].set_ylabel("S_t")
    fig.suptitle("Trajectoires simulees : rugosite croissante BS -> Heston -> rBergomi")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/trajectories_comparison.pdf")
    plt.close(fig)


def fig_implied_vol_smile(seed: int = 1) -> None:
    """Smile de vol implicite ATM court-terme, BS/Heston/rBergomi superposes
    (doit montrer l'explosion caracteristique du rBergomi, absente de Heston)."""
    S0, T, r, n_paths = 100.0, 0.1, 0.0, 200_000
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])

    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=r)
    rbergomi = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)

    n_steps = max(int(T * 50), 10)
    
    _, S_h, _ = heston.simulate(S0, v0=0.04, mu=r, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    _, S_r, _ = rbergomi.simulate_hybrid(S0=S0, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)

    def smile(S_T):
        ivs = []
        for K in strikes:
            price = np.exp(-r * T) * np.maximum(S_T - K, 0.0).mean()
            ivs.append(implied_vol(price, S0, K, r, T))
        return np.array(ivs)

    iv_bs = np.full(len(strikes), 0.2)  # BS : sigma constant = 0.2 par construction, pas de smile, pas de MC
    iv_h = smile(S_h[:, -1])
    iv_r = smile(S_r[:, -1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strikes, iv_bs, "o-", color=STYLE["bs"], label="Black-Scholes")
    ax.plot(strikes, iv_h, "o-", color=STYLE["heston"], label="Heston")
    ax.plot(strikes, iv_r, "o-", color=STYLE["rbergomi"], label="rBergomi (H=0.1)")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Vol implicite")
    ax.set_title(f"Smile de vol implicite, T={T} (maturite courte)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/implied_vol_smile.pdf")
    plt.close(fig)

def fig_pnl_distributions() -> None:
    """Distributions de P&L superposees (classique vs deep), une sous-figure
    par modele sous-jacent (roadmap Phase 13, resultats de la Phase 10)."""
    S0, K, T, r, sigma = 100.0, 100.0, 0.25, 0.0, 0.2
    n_steps, n_paths_train, n_epochs, n_paths_test = 20, 4_000, 300, 20_000
    mu = r

    pnl_bs_a = bs_delta_hedge_pnl(S0, mu, sigma, K, r, T, n_steps, n_paths_test, seed=TEST_SEED)
    ffn_bs = train_bs_hedger(S0, mu, sigma, K, r, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED)
    ffn_bs.eval()
    _, S_test_bs = simulate_gbm_paths_torch(S0, mu, sigma, T, n_steps, n_paths_test)
    from dhrv.models.black_scholes import bs_call_price
    p0_bs = bs_call_price(S0, K, r, sigma, T)
    pnl_bs_d = _eval_ffn_rollout(ffn_bs, S_test_bs, [], K, T, n_steps, n_paths_test) + p0_bs

    heston = None
    from dhrv.models.heston import HestonModel
    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=r)
    v0 = 0.04
    pnl_h_a = heston_delta_hedge_pnl(heston, S0, v0, mu, K, T, n_steps, n_paths_test, delta_type="heston_mv", seed=TEST_SEED)
    ffn_h = train_heston_hedger(heston, S0, v0, mu, K, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED)
    ffn_h.eval()
    _, S_test_h, v_test_h = simulate_heston_paths_torch(heston, S0, v0, mu, T, n_steps, n_paths_test, seed=TEST_SEED)
    p0_h = heston.call_price(np.array([S0]), np.array([v0]), K, T)[0]
    pnl_h_d = _eval_ffn_rollout(ffn_h, S_test_h, [v_test_h], K, T, n_steps, n_paths_test) + p0_h

    from dhrv.models.rbergomi import RBergomiModel
    rbergomi = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    pnl_r_a = rbergomi_proxy_hedge_pnl(rbergomi, S0, K, r, T, n_steps, n_paths_test, seed=TEST_SEED)
    ffn_r = train_rbergomi_hedger(rbergomi, S0, K, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED)
    ffn_r.eval()
    _, S_test_r, v_test_r = simulate_rbergomi_paths_torch(rbergomi, S0, T, n_steps, n_paths_test, seed=TEST_SEED)
    from dhrv.models.black_scholes import bs_call_price as _bscp
    p0_r = _bscp(S0, K, r, np.sqrt(rbergomi.xi0), T)
    pnl_r_d = _eval_ffn_rollout(ffn_r, S_test_r, [v_test_r], K, T, n_steps, n_paths_test) + p0_r

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    data = [
        ("Black-Scholes", pnl_bs_a, pnl_bs_d),
        ("Heston", pnl_h_a, pnl_h_d),
        ("rBergomi", pnl_r_a, pnl_r_d),
    ]
    for ax, (title, pnl_a, pnl_d) in zip(axes, data):
        ax.hist(pnl_a, bins=60, alpha=0.6, label="Classique", color="#185FA5", density=True)
        ax.hist(pnl_d, bins=60, alpha=0.6, label="Deep Hedging", color="#A32D2D", density=True)
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.legend()
    axes[0].set_ylabel("Densite")
    fig.suptitle("Distributions de P&L : classique vs Deep Hedging (sans couts, Phase 10)")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/pnl_distributions.pdf")
    plt.close(fig)


def fig_transaction_cost_gap() -> None:
    """Ecart de variance en fonction de kappa (roadmap Phase 11)."""
    kappas = [0.0, 0.0005, 0.001, 0.002, 0.005]
    results = run_transaction_cost_sweep(kappas, n_epochs=1500)
    gaps = performance_gap(results)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(gaps.keys()), list(gaps.values()), "o-", color="#3C3489")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("kappa (cout de transaction)")
    ax.set_ylabel("Var(classique) - Var(deep)  [positif = deep gagne]")
    ax.set_title("Ecart de performance vs cout de transaction")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/transaction_cost_gap.pdf")
    plt.close(fig)


def fig_robustness_degradation() -> None:
    """Degradation relative sous mauvaise specification de rho (roadmap Phase 12)."""
    rho_shocks = [-0.7, -0.5, -0.3, 0.0]
    results = run_robustness_sweep(rho_shocks)

    ref_classic = results[-0.7]["classic_mv_delta"]["std"]
    ref_deep = results[-0.7]["deep_hedging"]["std"]
    classic_ratios = [r["classic_mv_delta"]["std"] / ref_classic for r in results.values()]
    deep_ratios = [r["deep_hedging"]["std"] / ref_deep for r in results.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rho_shocks, classic_ratios, "o-", color="#185FA5", label="Classique (delta MV)")
    ax.plot(rho_shocks, deep_ratios, "o-", color="#A32D2D", label="Deep Hedging")
    ax.set_xlabel("rho reel (rho d'entrainement = -0.7)")
    ax.set_ylabel("Std relatif (vs reference rho=-0.7)")
    ax.set_title("Degradation sous mauvaise specification du levier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/robustness_degradation.pdf")
    plt.close(fig)


def fig_policy_heatmap() -> None:
    """Heatmap de la politique deep hedger vs delta BS analytique, en fonction
    de (S_t, t) -- roadmap Phase 13. Reutilise le deep hedger BS deja entraine
    (memes parametres que Phase 9.1/10)."""
    from dhrv.models.black_scholes import bs_call_delta

    S0, K, r, sigma, T = 100.0, 100.0, 0.0, 0.2, 0.25
    n_steps, n_paths_train, n_epochs = 20, 4_000, 300
    mu = r

    model = train_bs_hedger(S0, mu, sigma, K, r, T, n_steps, n_paths_train, n_epochs, seed=TRAIN_SEED)
    model.eval()

    S_grid = np.linspace(70, 130, 60)
    t_grid = np.linspace(0.0, T * 0.98, 40)  # evite tau=0 exact

    delta_deep = np.zeros((len(t_grid), len(S_grid)))
    delta_analytic = np.zeros((len(t_grid), len(S_grid)))

    with torch.no_grad():
        for i, t in enumerate(t_grid):
            tau = T - t
            S_t = torch.tensor(S_grid, dtype=torch.float32)
            log_moneyness = torch.log(S_t / K)
            tau_norm = torch.full_like(S_t, tau / T)
            # approximation : delta_prev=0 (politique "instantanee", independante
            # de l'historique de position -- simplification pour la visualisation,
            # le vrai rollout dependrait du chemin)
            delta_prev = torch.zeros_like(S_t)
            x = torch.stack([log_moneyness, tau_norm, delta_prev], dim=-1)
            delta_deep[i] = model(x).numpy()
            delta_analytic[i] = bs_call_delta(S_grid, K, r, sigma, tau)

    diff = delta_deep - delta_analytic

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title, cmap in zip(
        axes, [delta_analytic, delta_deep, diff],
        ["Delta BS analytique", "Politique Deep Hedging", "Ecart (deep - analytique)"],
        ["viridis", "viridis", "RdBu_r"],
    ):
        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap=cmap,
            extent=[S_grid.min(), S_grid.max(), t_grid.min(), t_grid.max()],
        )
        ax.set_xlabel("S_t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("t")
    fig.suptitle("Heatmap de la politique de couverture : Deep Hedging vs Delta BS analytique")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/policy_heatmap.pdf")
    plt.close(fig)

if __name__ == "__main__":
    fig_trajectories_side_by_side()
    fig_implied_vol_smile()
    fig_pnl_distributions()
    fig_transaction_cost_gap()
    fig_robustness_degradation()
    fig_policy_heatmap()
    print(f"Figures generees dans {FIGDIR}/")
    print(f"Figures 13.1 generees dans {FIGDIR}/")