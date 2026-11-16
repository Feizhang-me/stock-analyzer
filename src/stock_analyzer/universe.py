"""多标的批量分析与行业聚合。"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from .indicators import add_indicators
from .metrics import summarize

logger = logging.getLogger(__name__)


def analyze_universe(
    source,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    risk_free_rate: float = 0.02,
    refresh: bool = False,
) -> pd.DataFrame:
    """批量计算每只股票的区间表现。

    返回以 ts_code 为索引的 DataFrame，列包括 name、industry 与绩效指标。
    单只股票失败不会中断整体流程。
    """
    codes = list(dict.fromkeys(codes))
    if not codes:
        return pd.DataFrame()

    basic = source.stock_basic()
    basic = basic.set_index("ts_code")[["name", "industry"]]

    rows: list[dict] = []
    for i, code in enumerate(codes, start=1):
        try:
            df = source.daily(code, start_date, end_date, refresh=refresh)
            if df is None or df.empty or len(df) < 30:
                logger.warning("[%s/%s] %s 数据不足，跳过", i, len(codes), code)
                continue
            processed = add_indicators(df)
            metrics = summarize(processed["daily_return"].dropna(), risk_free_rate)
            metrics["ts_code"] = code
            metrics["last_close"] = float(processed["close"].iloc[-1])
            rows.append(metrics)
        except Exception as exc:  # 单只失败不影响整体
            logger.warning("[%s/%s] %s 分析失败: %s", i, len(codes), code, exc)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("ts_code")
    result = result.join(basic, how="left")
    result = result.sort_values("sharpe_ratio", ascending=False)
    return result[
        [
            "name",
            "industry",
            "last_close",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "n_days",
        ]
    ]


def industry_summary(performance: pd.DataFrame) -> pd.DataFrame:
    """按行业聚合等权平均表现。"""
    if performance.empty or "industry" not in performance.columns:
        return pd.DataFrame()
    valid = performance.dropna(subset=["industry"])
    if valid.empty:
        return pd.DataFrame()
    summary = valid.groupby("industry").agg(
        stock_count=("name", "count"),
        avg_annualized_return=("annualized_return", "mean"),
        median_annualized_return=("annualized_return", "median"),
        avg_sharpe=("sharpe_ratio", "mean"),
        avg_max_drawdown=("max_drawdown", "mean"),
        avg_win_rate=("win_rate", "mean"),
    )
    return summary.sort_values("avg_sharpe", ascending=False)
