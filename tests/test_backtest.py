import numpy as np
import pandas as pd
import pytest

from stock_analyzer.backtest import (
    ma_crossover_signal,
    rsi_mean_reversion_signal,
    run_backtest,
)


def make_price(close_values):
    index = pd.bdate_range("2022-01-03", periods=len(close_values), name="trade_date")
    close = pd.Series(close_values, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        }
    )


def test_no_lookahead_bias():
    price = make_price([100.0, 110.0, 121.0])
    signal = pd.Series([1.0, 1.0, 1.0], index=price.index)
    result = run_backtest(price, signal, commission=0.0, slippage=0.0)
    # t 日信号，t+1 日才生效：第一天收益必须为 0
    assert result.position.iloc[0] == 0.0
    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.returns.iloc[1] == pytest.approx(0.10)
    assert result.returns.iloc[2] == pytest.approx(0.10)


def test_costs_reduce_performance():
    price = make_price(np.linspace(100, 150, 60))
    signal = pd.Series(1.0, index=price.index)
    cheap = run_backtest(price, signal, commission=0.0, slippage=0.0)
    expensive = run_backtest(price, signal, commission=0.001, slippage=0.001)
    assert expensive.metrics["total_return"] < cheap.metrics["total_return"]


def test_ma_signal_requires_valid_windows():
    close = pd.Series(np.arange(50, dtype=float))
    with pytest.raises(ValueError):
        ma_crossover_signal(close, fast=20, slow=5)


def test_rsi_signal_is_binary_and_forward_filled():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))))
    signal = rsi_mean_reversion_signal(close, window=14, entry=30, exit_level=70)
    assert set(signal.unique()) <= {0.0, 1.0}
    assert signal.isna().sum() == 0


def test_benchmark_comparison_present():
    price = make_price(np.linspace(100, 120, 80))
    benchmark = pd.Series(np.linspace(4000, 4100, 80), index=price.index)
    signal = pd.Series(1.0, index=price.index)
    result = run_backtest(price, signal, benchmark_close=benchmark)
    assert result.benchmark_equity is not None
    assert result.benchmark_metrics is not None
    assert "sharpe_ratio" in result.benchmark_metrics
