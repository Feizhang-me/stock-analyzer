"""项目配置。

所有路径与参数集中在这里，避免散落在脚本中的硬编码。
Token 从环境变量读取，不写入源码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    """运行参数。"""

    token: str = ""
    data_dir: Path = Path("data")
    cache_path: Path = Path("data/cache.sqlite3")
    output_dir: Path = Path("output")
    start_date: str = "20200101"
    end_date: str = "20231231"
    benchmark: str = "000300.SH"
    risk_free_rate: float = 0.02
    commission: float = 0.0003
    slippage: float = 0.0002
    requests_per_minute: int = 200
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("STOCK_DATA_DIR", "data")
        return cls(
            token=os.getenv("TUSHARE_TOKEN", "").strip(),
            data_dir=data_dir,
            cache_path=_env_path("STOCK_CACHE_PATH", str(data_dir / "cache.sqlite3")),
            output_dir=_env_path("STOCK_OUTPUT_DIR", "output"),
            start_date=os.getenv("STOCK_START_DATE", cls.start_date),
            end_date=os.getenv("STOCK_END_DATE", cls.end_date),
            benchmark=os.getenv("STOCK_BENCHMARK", cls.benchmark),
            risk_free_rate=float(os.getenv("STOCK_RISK_FREE_RATE", cls.risk_free_rate)),
            commission=float(os.getenv("STOCK_COMMISSION", cls.commission)),
            slippage=float(os.getenv("STOCK_SLIPPAGE", cls.slippage)),
            requests_per_minute=int(os.getenv("STOCK_RPM", cls.requests_per_minute)),
            max_retries=int(os.getenv("STOCK_MAX_RETRIES", cls.max_retries)),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
