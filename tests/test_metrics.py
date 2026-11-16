import numpy as np
import pandas as pd
import pytest

from stock_analyzer.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    summarize,
    total_return,
)


def test_total_and_annualized_return():
    returns = pd.Series([0.001] * 252)
    assert total_return(returns) == pytest.approx(1.001**252 - 1)
    assert annualized_return(returns) == pytest.approx(1.001**252 - 1)


def test_annualized_volatility_constant_returns():
    returns = pd.Series([0.001] * 252)
    assert annualized_volatility(returns) == pytest.approx(0.0)
    assert sharpe_ratio(returns) == pytest.approx(0.0)


def test_max_drawdown_known_path():
    returns = pd.Series([0.10, -0.20, 0.05])
    dd = max_drawdown(returns)
    # 净值 1.10 -> 0.88，回撤 = 0.88/1.10 - 1 = -0.2
    assert dd.max_drawdown == pytest.approx(-0.2)
    assert dd.peak_date == 0
    assert dd.trough_date == 1


def test_summary_contains_expected_keys():
    returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    result = summarize(returns, risk_free_rate=0.02)
    expected = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "n_days",
    }
    assert expected <= set(result)
    assert result["max_drawdown"] <= 0
    assert 0 <= result["win_rate"] <= 1
    assert result["n_days"] == 300
