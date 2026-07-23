import numpy as np

from dhrv.evaluation.metrics import summary_stats


def test_summary_stats_on_known_distribution():
    """VaR95/CVaR95 sur une gaussienne standard doivent matcher les valeurs theoriques connues."""
    rng = np.random.default_rng(0)
    pnl = rng.standard_normal(500_000)

    stats = summary_stats(pnl, alpha=0.95)

    assert stats["mean"] == pytest_approx(0.0, abs=0.01)
    assert stats["std"] == pytest_approx(1.0, abs=0.01)
    assert stats["VaR95"] == pytest_approx(1.645, abs=0.05)  # quantile 95% de -N(0,1) = quantile 95% de N(0,1)


def pytest_approx(expected, abs):
    import pytest
    return pytest.approx(expected, abs=abs)