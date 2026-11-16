"""stock-analyzer: A 股行情分析、批量筛选与策略回测。"""

__version__ = "0.1.0"

from .backtest import BacktestResult, ma_crossover_signal, rsi_mean_reversion_signal, run_backtest
from .indicators import add_indicators
from .metrics import summarize

__all__ = [
    "BacktestResult",
    "add_indicators",
    "ma_crossover_signal",
    "rsi_mean_reversion_signal",
    "run_backtest",
    "summarize",
]
