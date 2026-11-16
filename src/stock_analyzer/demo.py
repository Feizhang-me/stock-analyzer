"""离线演示数据。

不依赖 Tushare token，用于快速验证指标、回测与报告生成流程。
生成的是几何随机游走，仅用于演示，不代表任何真实标的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    *,
    seed: int = 42,
    periods: int = 520,
    start_price: float = 20.0,
    annual_drift: float = 0.08,
    annual_volatility: float = 0.28,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-04", periods=periods)
    daily_drift = annual_drift / 252
    daily_vol = annual_volatility / np.sqrt(252)

    log_returns = rng.normal(daily_drift, daily_vol, periods)
    close = start_price * np.exp(np.cumsum(log_returns))

    prev_close = np.concatenate([[start_price], close[:-1]])
    gap = rng.normal(0.0, daily_vol * 0.25, periods)
    open_ = prev_close * (1.0 + gap)
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, periods)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, periods)))
    vol = rng.lognormal(mean=13.0, sigma=0.35, size=periods)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol": vol,
            "amount": vol * close,
        },
        index=pd.Index(index, name="trade_date"),
    )


def synthetic_benchmark(prices: pd.DataFrame, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.009, len(prices))
    values = 4000.0 * np.exp(np.cumsum(returns))
    return pd.Series(values, index=prices.index, name="benchmark")
