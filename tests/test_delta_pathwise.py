import numpy as np

from dhrv.models.rbergomi import RBergomiModel
from dhrv.hedging.delta_pathwise import rbergomi_pathwise_delta


def test_pathwise_delta_stable_across_epsilon():
    """Le delta estime doit converger vers une valeur stable sur une plage
    raisonnable d'epsilon (critere explicite de la roadmap, Phase 6)."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    S0, K, r, T = 100.0, 100.0, 0.0, 0.5
    n_steps, n_paths, seed = 25, 100_000, 42

    deltas = []
    for eps in [0.5, 1.0, 2.0]:
        d = rbergomi_pathwise_delta(model, S0, K, r, T, n_steps, n_paths, eps, seed)
        deltas.append(d)

    deltas = np.array(deltas)
    assert deltas.std() / deltas.mean() < 0.1  # variation relative faible entre eps
    assert 0.0 < deltas.mean() < 1.0  # delta de call plausible


def test_pathwise_delta_uses_common_random_numbers():
    """Verifie explicitement que les increments sont partages entre S0-eps et S0+eps
    (sinon la reduction de variance CRN n'opere pas)."""
    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    T, n_steps, n_paths, seed = 0.5, 25, 100, 42

    _, S_up, v_up = model.simulate_hybrid(S0=101.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    _, S_down, v_down = model.simulate_hybrid(S0=99.0, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)

    # v ne depend pas de S0 -> doit etre identique
    assert np.allclose(v_up, v_down)
    # ratio S_up/S_down doit etre constant sur toute la trajectoire (meme bruit multiplicatif)
    ratio = S_up / S_down
    assert np.allclose(ratio, ratio[:, [0]], rtol=1e-8)