import torch
import torch.nn as nn


class LSTMHedger(nn.Module):
    """Reseau recurrent, pertinent sous rBergomi (memoire longue, roadmap Phase 8).

    Entree attendue : (batch, n_steps, input_dim) -- toute la sequence de
    features de marche + position precedente concatenee a chaque pas.

    Sortie : (batch, n_steps), position a chaque instant.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)