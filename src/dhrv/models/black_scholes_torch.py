import torch


def simulate_gbm_paths_torch(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Version differentiable (tenseurs torch) de simulate_gbm_paths (Phase 2).

    Meme construction exacte (pas d'Euler) que la version numpy -- solution
    connue du GBM, reutilisee pour permettre l'autograd a travers la simulation.

    Returns
    -------
    t_grid : (n_steps+1,)
    S : (n_paths, n_steps+1), requires_grad via l'usage en aval (S0 peut etre un tenseur)
    """
    dt = T / n_steps
    dW = torch.randn(n_paths, n_steps, generator=generator) * (dt**0.5)
    W = torch.cumsum(dW, dim=1)
    t_grid = torch.linspace(0.0, T, n_steps + 1)

    log_S0 = torch.log(torch.as_tensor(float(S0)))
    log_increments = (mu - 0.5 * sigma**2) * t_grid[1:] + sigma * W
    log_S = log_S0 + log_increments
    log_S = torch.cat([torch.full((n_paths, 1), log_S0.item()), log_S], dim=1)

    return t_grid, torch.exp(log_S)