import torch
import torch.nn as nn


class FFNHedger(nn.Module):
    """Reseau feedforward partage par pas de temps.

    Entree attendue a chaque appel : (batch, input_dim), ou input_dim =
    nb de features de marche (ex. 2 pour (S_t, t) sous BS) + 1 (position
    precedente delta_{t-1}), conformement a la Phase 7.

    Sortie : (batch,), position (non bornee, pas de tanh -- pas de
    contrainte de levier pour ce projet).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)