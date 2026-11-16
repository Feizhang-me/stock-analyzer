"""轻量级策略回测。

只做日频、单标的、收盘价成交的简化回测，目的是验证策略逻辑，
而不是构建生产级交易系统。关键约束：
1. 信号在 t 日收盘后生成，t+1 日才计入收益，避免未来函数；
2. 计入双边手续费与滑点；
3. 与买入持有基准（默认沪深 300）对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import compute_rsi
from .metrics import Drawdown, max_drawdown, summarize


@dataclass
class BacktestResult:
    returns: pd.Series
    position: pd.Series
    equity: pd.Series
    turnover: pd.Series
    trades: int
    metrics: dict[str, float]
    benchmark_returns: pd.Series | None = None
    benchmark_equity: pd.Series | None = None
    benchmark_metrics: dict[str, float] | None = None
    drawdown: Drawdown = field(default_factory=lambda: Drawdown(0.0, None, None))


def ma_crossover_signal(close: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """双均线策略：快线在慢线上方时持有，否则空仓。"""
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("需满足 0 < fast < slow")
    close = pd.Series(close, dtype=float)
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = 0.0
    return signal


def rsi_mean_reversion_signal(
    close: pd.Series, window: int = 14, entry: float = 30.0, exit_level: float = 70.0
) -> pd.Series:
    """RSI 均值回归：RSI 低于 entry 买入，高于 exit_level 卖出。"""
    if not 0 < entry < exit_level < 100:
        raise ValueError("需满足 0 < entry < exit_level < 100")
    close = pd.Series(close, dtype=float)
    rsi = compute_rsi(close, window)
    position = pd.Series(np.nan, index=close.index, dtype=float)
    position[rsi < entry] = 1.0
    position[rsi > exit_level] = 0.0
    return position.ffill().fillna(0.0)


def _signal_to_position(signal: pd.Series, index: pd.Index) -> pd.Series:
    """把 t 日收盘信号转换为 t+1 日持仓，防止未来函数。"""
    signal = pd.Series(signal, dtype=float).reindex(index).fillna(0.0)
    return signal.clip(lower=0.0, upper=1.0).shift(1).fillna(0.0)


def run_backtest(
    price: pd.DataFrame,
    signal: pd.Series,
    *,
    commission: float = 0.0003,
    slippage: float = 0.0002,
    benchmark_close: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    periods: int = 252,
) -> BacktestResult:
    """执行回测并返回收益、净值与绩效指标。"""
    if price is None or price.empty or "close" not in price.columns:
        raise ValueError("price 必须包含非空 close 列")

    close = pd.Series(price["close"], dtype=float).sort_index()
    close = close[~close.index.duplicated(keep="last")]

    position = _signal_to_position(signal, close.index)
    asset_returns = close.pct_change(fill_method=None).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (commission + slippage)
    strategy_returns = position * asset_returns - cost
    equity = (1.0 + strategy_returns).cumprod()

    result = BacktestResult(
        returns=strategy_returns,
        position=position,
        equity=equity,
        turnover=turnover,
        trades=int((turnover > 1e-12).sum()),
        metrics=summarize(strategy_returns, risk_free_rate, periods),
        drawdown=max_drawdown(strategy_returns),
    )

    if benchmark_close is not None and len(benchmark_close) > 0:
        benchmark = pd.Series(benchmark_close, dtype=float).sort_index()
        benchmark = benchmark[~benchmark.index.duplicated(keep="last")]
        benchmark_returns = benchmark.pct_change(fill_method=None).reindex(close.index).fillna(0.0)
        result.benchmark_returns = benchmark_returns
        result.benchmark_equity = (1.0 + benchmark_returns).cumprod()
        result.benchmark_metrics = summarize(benchmark_returns, risk_free_rate, periods)

    return result


def compare_metrics(result: BacktestResult) -> pd.DataFrame:
    """把策略与基准指标排成一张对比表。"""
    if result.benchmark_metrics is None:
        return pd.DataFrame({"策略": result.metrics})
    return pd.DataFrame(
        {"策略": result.metrics, "基准": result.benchmark_metrics}
    )
