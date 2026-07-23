import torch

from dhrv.models.heston import HestonModel


def simulate_heston_paths_torch(
    model: HestonModel,
    S0: float,
    v0: float,
    mu: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simule des trajectoires Heston (QE scheme, deja valide en Phase 3, numpy)
    et les convertit en tenseurs torch.

    S et v sont des donnees de marche exogenes -- elles n'ont pas besoin de
    requires_grad (seul le graphe reseau doit rester differentiable, comme
    en Phase 9.1). Reutiliser le QE scheme numpy deja teste evite de
    redupliquer/redeboguer la logique de branchement psi<=psi_crit en torch.
    """
    t_grid_np, S_np, v_np = model.simulate(S0, v0, mu, T, n_steps, n_paths, seed=seed)
    t_grid = torch.from_numpy(t_grid_np).float()
    S = torch.from_numpy(S_np).float()
    v = torch.from_numpy(v_np).float()
    return t_grid, S, v