import torch

from dhrv.models.rbergomi import RBergomiModel


def simulate_rbergomi_paths_torch(
    model: RBergomiModel,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simule des trajectoires rBergomi (Hybrid scheme, deja valide Phase 5.2)
    et les convertit en tenseurs torch. Meme logique que heston_torch.py :
    S et v sont des donnees de marche exogenes, pas besoin de requires_grad.
    """
    t_grid_np, S_np, v_np = model.simulate_hybrid(S0=S0, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    t_grid = torch.from_numpy(t_grid_np).float()
    S = torch.from_numpy(S_np).float()
    v = torch.from_numpy(v_np).float()
    return t_grid, S, v