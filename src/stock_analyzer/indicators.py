"""技术指标计算。

输入为包含 open/high/low/close 列的日线 DataFrame，
输出为在原数据上追加指标列的新 DataFrame。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def _validate(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("行情数据为空")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI。平均涨跌幅使用 alpha=1/window 的指数平滑。"""
    if close is None or len(close) == 0:
        raise ValueError("收盘价序列为空")
    if window <= 0:
        raise ValueError("RSI 窗口必须为正整数")
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # 区间内只涨不跌时 RSI 定义为 100；完全不动定义为 50。
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range，用于衡量波动幅度。"""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def add_indicators(
    df: pd.DataFrame,
    ma_windows: tuple[int, ...] = (5, 10, 20, 60),
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    boll_window: int = 20,
    boll_std: float = 2.0,
    atr_window: int = 14,
) -> pd.DataFrame:
    """一次计算常用技术指标。

    新增列：
    ma{w}, daily_return, cumulative_return, volatility_20, rsi,
    macd, macd_signal, macd_hist, boll_mid/upper/lower, atr
    """
    _validate(df)
    if macd_fast >= macd_slow:
        raise ValueError("macd_fast 必须小于 macd_slow")

    out = df.copy()
    close = out["close"].astype(float)

    for window in ma_windows:
        if window <= 0:
            raise ValueError("均线窗口必须为正整数")
        out[f"ma{window}"] = close.rolling(window, min_periods=window).mean()

    out["daily_return"] = close.pct_change(fill_method=None)
    out["cumulative_return"] = (1.0 + out["daily_return"].fillna(0.0)).cumprod() - 1.0
    out["volatility_20"] = (
        out["daily_return"].rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    )

    out["rsi"] = compute_rsi(close, rsi_window)

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=macd_signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    boll_mid = close.rolling(boll_window, min_periods=boll_window).mean()
    boll_std_val = close.rolling(boll_window, min_periods=boll_window).std(ddof=0)
    out["boll_mid"] = boll_mid
    out["boll_upper"] = boll_mid + boll_std * boll_std_val
    out["boll_lower"] = boll_mid - boll_std * boll_std_val

    out["atr"] = compute_atr(out, atr_window)
    return out


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """返回 fast 上穿 slow 的布尔序列。"""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))
