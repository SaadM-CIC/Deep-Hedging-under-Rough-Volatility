import torch

from dhrv.hedging.ffn_hedger import FFNHedger
from dhrv.hedging.losses import entropic_risk
from dhrv.models.black_scholes_torch import simulate_gbm_paths_torch
from dhrv.models.heston import HestonModel
from dhrv.models.heston_torch import simulate_heston_paths_torch
from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.rbergomi_torch import simulate_rbergomi_paths_torch
from dhrv.hedging.lstm_hedger import LSTMHedger
def train_bs_hedger(
    S0: float,
    mu: float,
    sigma: float,
    K: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_epochs: int,
    lam: float = 1.0,
    lr: float = 1e-3,
    kappa: float = 0.0,
    seed: int | None = None,
) -> FFNHedger:
    """Entraine un FFNHedger sous BS, sans couts de transaction par defaut (kappa=0).

    A chaque epoch : nouveau batch de trajectoires, rollout de la politique
    pas a pas, P&L terminal differentiable, loss = entropic risk, backward.

    Features en entree du reseau a chaque pas : (S_t, tau, delta_prev),
    conforme a la Phase 7.

    kappa : cout de transaction proportionnel
    (kappa * |delta_new - delta_prev| * S_t), soustrait a chaque rebalancement.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = FFNHedger(input_dim=3, hidden_dim=32, n_hidden_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        t_grid, S = simulate_gbm_paths_torch(S0, mu, sigma, T, n_steps, n_paths)

        delta_prev = torch.zeros(n_paths)
        cash_gain = torch.zeros(n_paths)
        cost_total = torch.zeros(n_paths)

        for i in range(n_steps):
            tau = T - t_grid[i]
            S_t = S[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = (tau / T).expand(n_paths)
            x = torch.stack([log_moneyness, tau_norm, delta_prev], dim=-1)
            delta_new = model(x)

            cost_total = cost_total + kappa * torch.abs(delta_new - delta_prev) * S_t
            cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
            delta_prev = delta_new

        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = cash_gain - payoff - cost_total

        loss = entropic_risk(pnl, lam)
        

        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}, pnl_mean={pnl.mean().item():.4f}, "
                  f"pnl_std={pnl.std().item():.4f}")

    return model

def train_heston_hedger(
    heston_model: HestonModel,
    S0: float,
    v0: float,
    mu: float,
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_epochs: int,
    lam: float = 1.0,
    lr: float = 1e-3,
    kappa: float = 0.0,
    seed: int | None = None,
) -> FFNHedger:
    """Entraine un FFNHedger sous Heston, sans couts de transaction par defaut.

    Features : (log_moneyness, tau_norm, v_t, delta_prev) -- ajout de v_t
    par rapport a BS, conforme a la Phase 7 ((S_t, v_t, t) markovien sous Heston).

    kappa : cout de transaction proportionnel
    (kappa * |delta_new - delta_prev| * S_t), soustrait a chaque rebalancement.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = FFNHedger(input_dim=4, hidden_dim=32, n_hidden_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        path_seed = None if seed is None else seed + epoch
        t_grid, S, v = simulate_heston_paths_torch(
            heston_model, S0, v0, mu, T, n_steps, n_paths, seed=path_seed
        )

        delta_prev = torch.zeros(n_paths)
        cash_gain = torch.zeros(n_paths)
        cost_total = torch.zeros(n_paths)

        for i in range(n_steps):
            tau = T - t_grid[i]
            S_t, v_t = S[:, i], v[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = (tau / T).expand(n_paths)
            x = torch.stack([log_moneyness, tau_norm, v_t, delta_prev], dim=-1)
            delta_new = model(x)

            cost_total = cost_total + kappa * torch.abs(delta_new - delta_prev) * S_t
            cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
            delta_prev = delta_new

        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = cash_gain - payoff - cost_total
        loss = entropic_risk(pnl, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}, pnl_mean={pnl.mean().item():.4f}, "
                  f"pnl_std={pnl.std().item():.4f}")

    return model

def train_rbergomi_hedger(
    rbergomi_model: RBergomiModel,
    S0: float,
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_epochs: int,
    lam: float = 1.0,
    lr: float = 1e-3,
    kappa: float = 0.0,
    seed: int | None = None,
) -> FFNHedger:
    """Entraine un FFNHedger sous rBergomi, sans couts de transaction par defaut.

    FFN choisi comme premiere baseline (v_t condense une partie de la memoire
    du processus). Comparaison LSTM prevue en suite directe pour verifier
    empiriquement si le FFN est structurellement insuffisant (roadmap Phase 8).

    kappa : cout de transaction proportionnel
    (kappa * |delta_new - delta_prev| * S_t), soustrait a chaque rebalancement.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = FFNHedger(input_dim=4, hidden_dim=32, n_hidden_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        path_seed = None if seed is None else seed + epoch
        t_grid, S, v = simulate_rbergomi_paths_torch(
            rbergomi_model, S0, T, n_steps, n_paths, seed=path_seed
        )

        delta_prev = torch.zeros(n_paths)
        cash_gain = torch.zeros(n_paths)
        cost_total = torch.zeros(n_paths)

        for i in range(n_steps):
            tau = T - t_grid[i]
            S_t, v_t = S[:, i], v[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = (tau / T).expand(n_paths)
            x = torch.stack([log_moneyness, tau_norm, v_t, delta_prev], dim=-1)
            delta_new = model(x)

            cost_total = cost_total + kappa * torch.abs(delta_new - delta_prev) * S_t
            cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
            delta_prev = delta_new

        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = cash_gain - payoff - cost_total
        loss = entropic_risk(pnl, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}, pnl_mean={pnl.mean().item():.4f}, "
                  f"pnl_std={pnl.std().item():.4f}")

    return model

def train_rbergomi_hedger_lstm(
    rbergomi_model: RBergomiModel,
    S0: float,
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_epochs: int,
    lam: float = 1.0,
    lr: float = 1e-3,
    seed: int | None = None,
) -> LSTMHedger:
    """Entraine un LSTMHedger sous rBergomi, sans couts de transaction.

    Contrairement au FFN, tout le rollout est vectorise en un seul forward
    (pas de boucle python pas-a-pas) -- la memoire est portee par l'etat
    cache du LSTM, pas par une feature delta_prev explicite.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = LSTMHedger(input_dim=3, hidden_dim=32, n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        path_seed = None if seed is None else seed + epoch
        t_grid, S, v = simulate_rbergomi_paths_torch(
            rbergomi_model, S0, T, n_steps, n_paths, seed=path_seed
        )

        tau = (T - t_grid[:n_steps]) / T
        log_moneyness = torch.log(S[:, :n_steps] / K)
        v_t = v[:, :n_steps]
        tau_expanded = tau.unsqueeze(0).expand(n_paths, -1)

        x = torch.stack([log_moneyness, tau_expanded, v_t], dim=-1)
        deltas = model(x)

        increments = S[:, 1:] - S[:, :-1]
        cash_gain = (deltas * increments).sum(dim=1)

        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = cash_gain - payoff
        loss = entropic_risk(pnl, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}, pnl_mean={pnl.mean().item():.4f}, "
                  f"pnl_std={pnl.std().item():.4f}")

    return model