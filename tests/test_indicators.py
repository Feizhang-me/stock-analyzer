import numpy as np
import pandas as pd
import pytest

from stock_analyzer.indicators import add_indicators, compute_rsi, crossover


def make_prices(n: int = 120) -> pd.DataFrame:
    close = np.arange(1.0, n + 1.0)
    index = pd.bdate_range("2022-01-03", periods=n, name="trade_date")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "vol": np.full(n, 1000.0),
        },
        index=index,
    )


def test_moving_average_values():
    df = add_indicators(make_prices())
    # 第 5 天的 MA5 = (1+2+3+4+5)/5 = 3
    assert df["ma5"].iloc[4] == pytest.approx(3.0)
    assert df["ma20"].iloc[19] == pytest.approx(10.5)
    assert df["ma5"].iloc[:4].isna().all()


def test_rsi_bounds_and_trend():
    df = add_indicators(make_prices())
    rsi = df["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    # 单调上涨序列，RSI 应为 100
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_bollinger_bands_order():
    df = add_indicators(make_prices())
    valid = df.dropna(subset=["boll_upper", "boll_mid", "boll_lower"])
    assert (valid["boll_upper"] >= valid["boll_mid"]).all()
    assert (valid["boll_mid"] >= valid["boll_lower"]).all()


def test_macd_hist_matches_difference():
    df = add_indicators(make_prices())
    diff = df["macd"] - df["macd_signal"]
    assert np.allclose(diff, df["macd_hist"])


def test_crossover_detection():
    fast = pd.Series([1, 2, 3, 4, 3], dtype=float)
    slow = pd.Series([2, 2, 2, 2, 2], dtype=float)
    result = crossover(fast, slow)
    assert result.tolist() == [False, False, True, False, False]


def test_empty_input_raises():
    with pytest.raises(ValueError):
        compute_rsi(pd.Series(dtype=float))
    with pytest.raises(ValueError):
        add_indicators(pd.DataFrame())
