"""离线演示：不调用 Tushare，直接生成一份示例报告。

用法：
    python examples/synthetic_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_analyzer.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo"]))
