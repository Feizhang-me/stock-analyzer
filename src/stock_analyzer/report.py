"""Plotly 图表与 HTML 报告。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .backtest import BacktestResult

PALETTE = {
    "primary": "#167C80",
    "secondary": "#F2643D",
    "neutral": "#6B7280",
    "up": "#D64545",
    "down": "#2F9E6E",
    "grid": "#E5E7EB",
}


def price_technical_figure(df: pd.DataFrame, title: str) -> go.Figure:
    """价格 / 均线 / 布林带、成交量、MACD、RSI 四联图。"""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.46, 0.16, 0.19, 0.19],
        subplot_titles=("价格与均线", "成交量", "MACD", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K 线",
            increasing_line_color=PALETTE["up"],
            decreasing_line_color=PALETTE["down"],
        ),
        row=1,
        col=1,
    )

    for window, color in ((5, "#F2643D"), (20, "#167C80"), (60, "#8B5CF6")):
        col = f"ma{window}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=f"MA{window}", line=dict(color=color, width=1)),
                row=1,
                col=1,
            )

    if {"boll_upper", "boll_lower"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["boll_upper"],
                name="布林上轨",
                line=dict(color=PALETTE["neutral"], width=0.8, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["boll_lower"],
                name="布林下轨",
                line=dict(color=PALETTE["neutral"], width=0.8, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(107,114,128,0.08)",
            ),
            row=1,
            col=1,
        )

    if "vol" in df.columns:
        colors = [
            PALETTE["up"] if close >= open_ else PALETTE["down"]
            for open_, close in zip(df["open"], df["close"], strict=False)
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["vol"], name="成交量", marker_color=colors, opacity=0.7),
            row=2,
            col=1,
        )

    if {"macd", "macd_signal", "macd_hist"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color=PALETTE["primary"], width=1)),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd_signal"],
                name="信号线",
                line=dict(color=PALETTE["secondary"], width=1),
            ),
            row=3,
            col=1,
        )
        hist_colors = [PALETTE["up"] if v >= 0 else PALETTE["down"] for v in df["macd_hist"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["macd_hist"], name="MACD 柱", marker_color=hist_colors, opacity=0.6),
            row=3,
            col=1,
        )

    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#8B5CF6", width=1)),
            row=4,
            col=1,
        )
        fig.add_hline(y=70, line=dict(color=PALETTE["up"], width=1, dash="dash"), row=4, col=1)
        fig.add_hline(y=30, line=dict(color=PALETTE["down"], width=1, dash="dash"), row=4, col=1)
        fig.update_yaxes(range=[0, 100], row=4, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        margin=dict(l=50, r=30, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        bargap=0.1,
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def equity_figure(result: BacktestResult, title: str) -> go.Figure:
    """策略净值与基准净值对比。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index,
            y=result.equity,
            name="策略净值",
            line=dict(color=PALETTE["primary"], width=1.6),
        )
    )
    if result.benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=result.benchmark_equity.index,
                y=result.benchmark_equity,
                name="基准净值",
                line=dict(color=PALETTE["neutral"], width=1.2, dash="dash"),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=460,
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title="净值（初始 = 1）")
    return fig


def _format_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype.kind in "fc":
            if col in {
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "max_drawdown",
                "win_rate",
                "avg_annualized_return",
                "median_annualized_return",
                "avg_max_drawdown",
                "avg_win_rate",
            }:
                formatted[col] = formatted[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
            else:
                formatted[col] = formatted[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    return formatted.to_html(classes="summary", border=0, escape=True, na_rep="-")


def write_report(
    path: str | Path,
    *,
    title: str,
    figures: list[go.Figure],
    summary: pd.DataFrame | None = None,
    notes: str | None = None,
) -> Path:
    """把图表和表格写成一个自包含的 HTML 文件（Plotly 从 CDN 加载）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figures_html = "\n".join(
        pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False})
        for fig in figures
    )
    table_html = _format_table(summary) if summary is not None and not summary.empty else ""
    notes_html = f"<p class='notes'>{html.escape(notes)}</p>" if notes else ""

    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; padding: 32px 24px 56px; background: #F9FAFB;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
         color: #1F2937; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin: 0 0 6px; }}
  .sub {{ color: #6B7280; font-size: 13px; margin-bottom: 24px; }}
  .card {{ background: #fff; border: 1px solid #E5E7EB; border-radius: 12px;
           padding: 18px 20px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  table.summary {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.summary th {{ text-align: left; background: #F3F4F6; color: #374151;
                      padding: 8px 10px; font-weight: 600; }}
  table.summary td {{ padding: 7px 10px; border-top: 1px solid #E5E7EB; }}
  table.summary tr:nth-child(even) td {{ background: #FBFBFC; }}
  .notes {{ color: #6B7280; font-size: 12px; line-height: 1.7; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <div class="sub">stock-analyzer · 自动生成报告 · 数据仅用于研究，不构成投资建议</div>
  {f'<div class="card"><h3>绩效汇总</h3>{table_html}</div>' if table_html else ''}
  <div class="card">{figures_html}</div>
  {f'<div class="card">{notes_html}</div>' if notes_html else ''}
</div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path
