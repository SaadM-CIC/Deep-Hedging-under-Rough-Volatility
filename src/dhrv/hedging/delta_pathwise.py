import numpy as np

from dhrv.models.rbergomi import RBergomiModel


def rbergomi_pathwise_delta(
    model: RBergomiModel,
    S0: float,
    K: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    eps: float,
    seed: int,
) -> float:
    """Delta par differences finies pathwise (common random numbers).

    CRN obtenu gratuitement : simulate_hybrid tire ses aleas via un rng interne
    seede en tout debut de fonction, avant toute utilisation de S0 — donc appeler
    avec la meme seed pour S0-eps et S0+eps produit exactement les memes increments
    (dW1, dW_perp), seule la condition initiale du prix differe.
    """
    _, S_up, _ = model.simulate_hybrid(S0=S0 + eps, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    _, S_down, _ = model.simulate_hybrid(S0=S0 - eps, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)

    payoff_up = np.maximum(S_up[:, -1] - K, 0.0)
    payoff_down = np.maximum(S_down[:, -1] - K, 0.0)

    price_up = np.exp(-r * T) * payoff_up.mean()
    price_down = np.exp(-r * T) * payoff_down.mean()

    return (price_up - price_down) / (2 * eps)