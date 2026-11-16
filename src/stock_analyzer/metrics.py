"""风险收益指标。

统一以日收益率序列为输入，年化按 252 个交易日计算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean_returns(returns: pd.Series) -> pd.Series:
    if returns is None:
        return pd.Series(dtype=float)
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def daily_returns(close: pd.Series) -> pd.Series:
    return pd.Series(close, dtype=float).pct_change(fill_method=None).dropna()


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    returns = _clean_returns(returns)
    return initial * (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() ** (periods / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = _clean_returns(returns)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods: int = TRADING_DAYS
) -> float:
    returns = _clean_returns(returns)
    if len(returns) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = returns - rf_daily
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods: int = TRADING_DAYS
) -> float:
    returns = _clean_returns(returns)
    if len(returns) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if not np.isfinite(downside_std) or downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods))


@dataclass(frozen=True)
class Drawdown:
    max_drawdown: float
    peak_date: object | None
    trough_date: object | None


def max_drawdown(returns: pd.Series) -> Drawdown:
    """最大回撤（负数）。peak_date 为回撤前高点，trough_date 为低点。"""
    returns = _clean_returns(returns)
    if returns.empty:
        return Drawdown(0.0, None, None)
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough_date = drawdown.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()
    return Drawdown(float(drawdown.min()), peak_date, trough_date)


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns).max_drawdown
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def summarize(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = TRADING_DAYS,
) -> dict[str, float]:
    """汇总常用绩效指标。"""
    returns = _clean_returns(returns)
    mdd = max_drawdown(returns)
    return {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns, periods),
        "annualized_volatility": annualized_volatility(returns, periods),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods),
        "max_drawdown": mdd.max_drawdown,
        "calmar_ratio": calmar_ratio(returns, periods),
        "win_rate": win_rate(returns),
        "n_days": float(len(returns)),
    }
