"""命令行入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .backtest import (
    BacktestResult,
    compare_metrics,
    ma_crossover_signal,
    rsi_mean_reversion_signal,
    run_backtest,
)
from .config import Settings
from .data import DataSourceError, TushareDataSource
from .demo import synthetic_benchmark, synthetic_prices
from .indicators import add_indicators
from .metrics import summarize
from .report import equity_figure, price_technical_figure, write_report
from .universe import analyze_universe, industry_summary

METRIC_LABELS = {
    "total_return": "累计收益",
    "annualized_return": "年化收益",
    "annualized_volatility": "年化波动率",
    "sharpe_ratio": "夏普比率",
    "sortino_ratio": "索提诺比率",
    "max_drawdown": "最大回撤",
    "calmar_ratio": "卡玛比率",
    "win_rate": "日胜率",
    "n_days": "交易日数",
}
PERCENT_METRICS = {
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "win_rate",
    "avg_annualized_return",
    "median_annualized_return",
    "avg_max_drawdown",
    "avg_win_rate",
}


def normalize_code(code: str) -> str:
    code = code.strip().upper()
    if "." in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("0", "2", "3")):
            return f"{code}.SZ"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
    return code


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        label = METRIC_LABELS.get(key, key)
        if key in PERCENT_METRICS:
            print(f"{label:<8}: {value:>10.2%}")
        elif key == "n_days":
            print(f"{label:<8}: {int(value):>10d}")
        else:
            print(f"{label:<8}: {value:>10.3f}")


def _print_comparison(result: BacktestResult) -> None:
    table = compare_metrics(result).drop(index="n_days", errors="ignore")
    table.index = [METRIC_LABELS.get(i, i) for i in table.index]
    pct_labels = {METRIC_LABELS[k] for k in PERCENT_METRICS}
    columns = list(table.columns)
    print("\n=== 策略 vs 基准 ===")
    print(f"{'指标':<10}" + "".join(f"{col:>12}" for col in columns))
    for idx, row in table.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                cells.append(f"{'-':>12}")
            elif idx in pct_labels:
                cells.append(f"{val:>11.2%}")
            else:
                cells.append(f"{val:>12.3f}")
        print(f"{idx:<10}" + "".join(cells))


def _make_source(settings: Settings) -> TushareDataSource:
    if not settings.token:
        raise DataSourceError(
            "未检测到 TUSHARE_TOKEN。请先设置环境变量，例如：\n"
            '  PowerShell: $env:TUSHARE_TOKEN="你的token"\n'
            "也可以先运行 `python main.py demo` 使用离线示例数据。"
        )
    return TushareDataSource(
        settings.token,
        settings.cache_path,
        requests_per_minute=settings.requests_per_minute,
        max_retries=settings.max_retries,
    )


def cmd_analyze(args: argparse.Namespace, settings: Settings) -> int:
    source = _make_source(settings)
    code = normalize_code(args.code)
    df = source.daily(code, args.start, args.end, refresh=args.refresh)
    if df.empty:
        print(f"未获取到 {code} 的行情数据，请检查代码或日期范围。")
        return 1

    processed = add_indicators(df)
    metrics = summarize(processed["daily_return"].dropna(), settings.risk_free_rate)
    name = source.stock_name(code)
    print(f"\n标的: {name} ({code})  区间: {args.start} - {args.end}")
    print(f"最新收盘价: {processed['close'].iloc[-1]:.2f}")
    _print_metrics(f"{name} 区间表现", metrics)

    report_path = Path(args.report) if args.report else settings.output_dir / f"{code}_analysis.html"
    fig = price_technical_figure(processed, f"{name}（{code}）技术分析")
    summary = pd.DataFrame([metrics])
    summary.index = [f"{name}（{code}）"]
    summary.columns = [METRIC_LABELS.get(c, c) for c in summary.columns]
    write_report(
        report_path,
        title=f"{name}（{code}）技术分析报告",
        figures=[fig],
        summary=summary,
        notes="数据来源：Tushare Pro。技术指标仅用于研究，不构成投资建议。",
    )
    print(f"\n报告已生成: {report_path.resolve()}")
    return 0


def cmd_backtest(args: argparse.Namespace, settings: Settings) -> int:
    source = _make_source(settings)
    code = normalize_code(args.code)
    df = source.daily(code, args.start, args.end, refresh=args.refresh)
    if df.empty:
        print(f"未获取到 {code} 的行情数据。")
        return 1

    processed = add_indicators(df)
    if args.strategy == "ma":
        signal = ma_crossover_signal(processed["close"], args.fast, args.slow)
        strategy_name = f"MA {args.fast}/{args.slow} 双均线"
    else:
        signal = rsi_mean_reversion_signal(
            processed["close"], args.rsi_window, args.entry, args.exit_level
        )
        strategy_name = f"RSI({args.rsi_window}) 均值回归"

    benchmark_close = None
    try:
        benchmark_df = source.index_daily(
            args.benchmark, args.start, args.end, refresh=args.refresh
        )
        if not benchmark_df.empty:
            benchmark_close = benchmark_df["close"]
    except DataSourceError as exc:
        logging.warning("基准指数获取失败，回测将不包含基准对比: %s", exc)

    result = run_backtest(
        processed,
        signal,
        commission=settings.commission,
        slippage=settings.slippage,
        benchmark_close=benchmark_close,
        risk_free_rate=settings.risk_free_rate,
    )

    name = source.stock_name(code)
    print(f"\n标的: {name} ({code})  策略: {strategy_name}")
    print(f"区间: {args.start} - {args.end}  交易次数: {result.trades}")
    _print_metrics(f"{strategy_name} 绩效", result.metrics)
    if result.benchmark_metrics:
        _print_metrics(f"基准 {args.benchmark}", result.benchmark_metrics)
        _print_comparison(result)

    report_path = Path(args.report) if args.report else settings.output_dir / f"{code}_backtest.html"
    figures = [equity_figure(result, f"{name} {strategy_name} 净值曲线")]
    summary = compare_metrics(result)
    summary.index = [METRIC_LABELS.get(i, i) for i in summary.index]
    write_report(
        report_path,
        title=f"{name}（{code}）{strategy_name} 回测报告",
        figures=figures,
        summary=summary,
        notes=(
            f"手续费 {settings.commission:.4%} + 滑点 {settings.slippage:.4%}，"
            "信号次日生效；结果不含分红再投资，不构成投资建议。"
        ),
    )
    print(f"\n报告已生成: {report_path.resolve()}")
    return 0


def cmd_screen(args: argparse.Namespace, settings: Settings) -> int:
    source = _make_source(settings)
    basic = source.stock_basic()
    matched = basic[basic["industry"].fillna("").str.contains(args.industry, regex=False)]
    if matched.empty:
        print(f"未找到行业包含“{args.industry}”的股票。")
        return 1

    codes = matched["ts_code"].head(args.limit).tolist()
    print(f"行业: {args.industry}  候选 {len(codes)} 只，开始批量分析（每只需一次 API 调用）...")
    performance = analyze_universe(
        source,
        codes,
        args.start,
        args.end,
        risk_free_rate=settings.risk_free_rate,
        refresh=args.refresh,
    )
    if performance.empty:
        print("批量分析未得到有效结果。")
        return 1

    print("\n=== 个股表现（按夏普比率排序） ===")
    display = performance.copy()
    for col in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown", "win_rate"]:
        display[col] = display[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    display["sharpe_ratio"] = display["sharpe_ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(display.to_string())

    summary = industry_summary(performance)
    if not summary.empty:
        print("\n=== 行业聚合 ===")
        print(summary.head(10).to_string(float_format=lambda x: f"{x:.3f}"))

    csv_path = settings.output_dir / f"screen_{args.industry}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(csv_path, encoding="utf-8-sig")
    print(f"\n结果已保存: {csv_path.resolve()}")
    return 0


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    prices = synthetic_prices(seed=args.seed)
    benchmark = synthetic_benchmark(prices)
    processed = add_indicators(prices)
    metrics = summarize(processed["daily_return"].dropna(), settings.risk_free_rate)

    signal = ma_crossover_signal(processed["close"], 5, 20)
    result = run_backtest(
        processed,
        signal,
        commission=settings.commission,
        slippage=settings.slippage,
        benchmark_close=benchmark,
        risk_free_rate=settings.risk_free_rate,
    )

    print("使用合成随机游走数据（非真实行情）演示完整流程。")
    _print_metrics("演示标的区间表现", metrics)
    _print_metrics("MA 5/20 双均线绩效", result.metrics)

    report_path = settings.output_dir / "demo_report.html"
    figures = [
        price_technical_figure(processed, "演示标的（合成数据）技术分析"),
        equity_figure(result, "MA 5/20 双均线策略净值"),
    ]
    summary = compare_metrics(result)
    summary.index = [METRIC_LABELS.get(i, i) for i in summary.index]
    write_report(
        report_path,
        title="stock-analyzer 离线演示报告",
        figures=figures,
        summary=summary,
        notes="数据由固定随机种子生成，仅用于验证代码流程，不代表任何真实标的。",
    )
    print(f"\n演示报告已生成: {report_path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-analyzer",
        description="基于 Tushare 的 A 股行情分析、批量筛选与策略回测工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
        p.add_argument("--end", default="20231231", help="结束日期 YYYYMMDD")
        p.add_argument("--refresh", action="store_true", help="忽略缓存，重新拉取")

    analyze = sub.add_parser("analyze", help="单只股票的技术指标分析")
    analyze.add_argument("--code", required=True, help="股票代码，如 000338.SZ 或 000338")
    analyze.add_argument("--report", help="HTML 报告输出路径")
    add_common(analyze)
    analyze.set_defaults(func=cmd_analyze)

    backtest = sub.add_parser("backtest", help="策略回测并与基准对比")
    backtest.add_argument("--code", required=True)
    backtest.add_argument("--strategy", choices=["ma", "rsi"], default="ma")
    backtest.add_argument("--fast", type=int, default=5, help="MA 快线窗口")
    backtest.add_argument("--slow", type=int, default=20, help="MA 慢线窗口")
    backtest.add_argument("--rsi-window", type=int, default=14)
    backtest.add_argument("--entry", type=float, default=30.0, help="RSI 买入阈值")
    backtest.add_argument("--exit-level", type=float, default=70.0, help="RSI 卖出阈值")
    backtest.add_argument("--benchmark", default="000300.SH")
    backtest.add_argument("--report", help="HTML 报告输出路径")
    add_common(backtest)
    backtest.set_defaults(func=cmd_backtest)

    screen = sub.add_parser("screen", help="按行业批量筛选并排序")
    screen.add_argument("--industry", required=True, help="行业关键词，如 银行、白酒")
    screen.add_argument("--limit", type=int, default=15, help="最多分析多少只")
    add_common(screen)
    screen.set_defaults(func=cmd_screen)

    demo = sub.add_parser("demo", help="用合成数据离线演示")
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    settings.ensure_dirs()
    try:
        return int(args.func(args, settings))
    except DataSourceError as exc:
        print(f"数据源错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
