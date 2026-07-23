import torch

from dhrv.hedging.train import train_rbergomi_hedger_lstm
from dhrv.models.rbergomi import RBergomiModel
from dhrv.models.rbergomi_torch import simulate_rbergomi_paths_torch


def _evaluate_lstm(model, rbergomi_model, S0, K, T, n_steps, n_eval, seed):
    model.eval()
    with torch.no_grad():
        t_grid, S, v = simulate_rbergomi_paths_torch(rbergomi_model, S0, T, n_steps, n_eval, seed=seed)
        tau = (T - t_grid[:n_steps]) / T
        log_moneyness = torch.log(S[:, :n_steps] / K)
        v_t = v[:, :n_steps]
        tau_expanded = tau.unsqueeze(0).expand(n_eval, -1)
        x = torch.stack([log_moneyness, tau_expanded, v_t], dim=-1)
        deltas = model(x)
        increments = S[:, 1:] - S[:, :-1]
        cash_gain = (deltas * increments).sum(dim=1)
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        return (cash_gain - payoff).numpy()


def test_lstm_vs_ffn_under_rbergomi():
    """Compare LSTM et FFN sous rBergomi (meme protocole, seeds identiques).
    Teste empiriquement l'hypothese roadmap Phase 8 : le FFN peut etre
    structurellement insuffisant sous un processus non-markovien -- a verifier,
    pas a supposer. Les deux resultats (LSTM meilleur ou pas) sont valides
    scientifiquement et doivent etre documentes tels quels.
    """
    from dhrv.hedging.train import train_rbergomi_hedger  # FFN, deja valide

    model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    S0, K, T = 100.0, 100.0, 0.25
    n_steps, n_paths, n_epochs = 20, 4_000, 300
    n_eval = 20_000

    ffn = train_rbergomi_hedger(model, S0, K, T, n_steps, n_paths, n_epochs, lam=1.0, lr=1e-3, seed=1)
    lstm = train_rbergomi_hedger_lstm(model, S0, K, T, n_steps, n_paths, n_epochs, lam=1.0, lr=1e-3, seed=1)

    pnl_ffn = _evaluate_lstm.__wrapped__ if False else None  # placeholder, non utilise

    # evaluation FFN (meme logique que test_train_rbergomi.py)
    ffn.eval()
    with torch.no_grad():
        t_grid, S, v = simulate_rbergomi_paths_torch(model, S0, T, n_steps, n_eval, seed=500)
        delta_prev = torch.zeros(n_eval)
        cash_gain = torch.zeros(n_eval)
        for i in range(n_steps):
            tau = T - t_grid[i]
            S_t, v_t = S[:, i], v[:, i]
            log_moneyness = torch.log(S_t / K)
            tau_norm = (tau / T).expand(n_eval)
            x = torch.stack([log_moneyness, tau_norm, v_t, delta_prev], dim=-1)
            delta_new = ffn(x)
            cash_gain = cash_gain + delta_new * (S[:, i + 1] - S[:, i])
            delta_prev = delta_new
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl_ffn = (cash_gain - payoff).numpy()

    pnl_lstm = _evaluate_lstm(lstm, model, S0, K, T, n_steps, n_eval, seed=500)

    print(f"std FFN = {pnl_ffn.std():.4f}")
    print(f"std LSTM = {pnl_lstm.std():.4f}")

    # sanity check large uniquement -- le LSTM ne doit pas exploser
    assert pnl_lstm.std() < 3 * pnl_ffn.std()