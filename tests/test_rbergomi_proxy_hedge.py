import numpy as np

from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.heston import HestonModel
from dhrv.hedging.delta_rbergomi_proxy import rbergomi_proxy_hedge_pnl
from dhrv.hedging.delta_heston import heston_delta_hedge_pnl


def test_rbergomi_proxy_pnl_is_finite_and_reasonable():
    """Sanity check : le P&L ne doit pas exploser ni contenir de NaN."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    S0, K, r, T = 100.0, 100.0, 0.0, 0.5

    pnl = rbergomi_proxy_hedge_pnl(model, S0, K, r, T, n_rebal=50, n_paths=20_000, seed=42)

    assert np.all(np.isfinite(pnl))
    assert abs(pnl.mean()) < 5.0  # pas de biais grossier


def test_rbergomi_hedging_degrades_vs_heston():
    """Var(P&L) sous rBergomi (proxy) doit etre superieure a celle sous Heston
    avec delta correct (heston_mv) -- degradation attendue, documentee par la
    roadmap comme resultat scientifique, pas un echec d'implementation."""
    rbergomi = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.0)
    S0, v0, mu, K, T = 100.0, 0.04, 0.0, 100.0, 0.5
    n_rebal, n_paths = 50, 20_000

    pnl_rbergomi = rbergomi_proxy_hedge_pnl(rbergomi, S0, K, 0.0, T, n_rebal, n_paths, seed=42)
    pnl_heston = heston_delta_hedge_pnl(
        heston, S0, v0, mu, K, T, n_rebal, n_paths, delta_type="heston_mv", seed=42
    )

    print(f"var rbergomi (proxy) = {pnl_rbergomi.var():.4f}")
    print(f"var heston (mv delta) = {pnl_heston.var():.4f}")
    assert pnl_rbergomi.var() > pnl_heston.var()