import torch

from dhrv.hedging.ffn_hedger import FFNHedger
from dhrv.hedging.lstm_hedger import LSTMHedger


def test_ffn_hedger_output_shape():
    """Un appel FFN a un instant t donne : (batch, input_dim) -> (batch,)."""
    batch, input_dim = 64, 3  # ex: (S_t, t, delta_prev) sous BS
    model = FFNHedger(input_dim=input_dim, hidden_dim=16, n_hidden_layers=2)

    x = torch.randn(batch, input_dim)
    out = model(x)

    assert out.shape == (batch,)
    assert torch.isfinite(out).all()


def test_ffn_hedger_forward_pass_no_error_multiple_steps():
    """Simule un rollout factice de n_steps appels sequentiels (comme en Phase 9)."""
    batch, input_dim, n_steps = 32, 3, 10
    model = FFNHedger(input_dim=input_dim)

    delta_prev = torch.zeros(batch)
    for _ in range(n_steps):
        market_features = torch.randn(batch, input_dim - 1)
        x = torch.cat([market_features, delta_prev.unsqueeze(-1)], dim=-1)
        delta_prev = model(x)

    assert delta_prev.shape == (batch,)


def test_lstm_hedger_output_shape():
    """LSTM traite toute la sequence d'un coup : (batch, n_steps, input_dim) -> (batch, n_steps)."""
    batch, n_steps, input_dim = 32, 20, 3  # ex: fenetre glissante sous rBergomi
    model = LSTMHedger(input_dim=input_dim, hidden_dim=16)

    x = torch.randn(batch, n_steps, input_dim)
    out = model(x)

    assert out.shape == (batch, n_steps)
    assert torch.isfinite(out).all()


def test_lstm_hedger_handles_variable_batch_size():
    """Verifie l'absence d'hypothese cachee sur une taille de batch fixe."""
    input_dim, n_steps = 3, 15
    model = LSTMHedger(input_dim=input_dim)

    for batch in [1, 8, 100]:
        x = torch.randn(batch, n_steps, input_dim)
        out = model(x)
        assert out.shape == (batch, n_steps)