# stock-analyzer：A 股行情分析与策略回测

把最初单文件、面向演示的股票分析脚本，重构为一个可维护的数据分析项目：
从 Tushare Pro 拉取行情与财务数据，本地缓存后计算技术指标和风险收益指标，
支持多标的批量筛选、双均线 / RSI 策略回测，并生成交互式 HTML 报告。

> 仅用于数据分析学习与研究，不构成任何投资建议。

## 功能

- **数据层**
  - 封装 Tushare Pro 的股票列表、日线、指数日线、财务指标接口
  - SQLite 本地缓存，日线按日期区间**增量更新**，重复运行不再重复请求
  - 请求限流 + 指数退避重试；接口不可用时自动回退到已有缓存
- **指标层**
  - MA(5/10/20/60)、RSI（Wilder）、MACD、布林带、ATR、年化波动率
  - 累计收益率与日收益率
- **绩效层**
  - 累计收益、年化收益、年化波动率、夏普比率、索提诺比率
  - 最大回撤（含峰值 / 谷值日期）、卡玛比率、日胜率
- **策略回测**
  - 双均线（MA 金叉持有 / 死叉空仓）
  - RSI 均值回归（超卖买入、超买卖出）
  - 信号次日生效，避免未来函数；计入手续费与滑点
  - 与沪深 300 等基准指数对比净值与绩效
- **批量分析**
  - 按行业筛选股票，批量计算区间表现并按夏普比率排序
  - 行业层面等权聚合，输出 CSV（UTF-8 BOM，Excel 可直接打开）
- **报告**
  - Plotly 交互式 HTML：K 线 / 均线 / 布林带 / 成交量 / MACD / RSI / 净值曲线
  - 无需浏览器插件，单文件即可分享
- **工程化**
  - 配置集中管理，Token 从环境变量读取
  - pytest 单元测试覆盖指标、绩效与回测逻辑

## 技术栈

Python 3.10+ · pandas · NumPy · Tushare Pro · Plotly · SQLite · pytest

## 目录结构

```
股票分析项目/
├── main.py                     # 命令行入口
├── pyproject.toml              # 依赖与打包配置
├── requirements.txt
├── src/stock_analyzer/
│   ├── config.py               # 环境变量与运行参数
│   ├── data.py                 # Tushare 访问、限流重试、SQLite 增量缓存
│   ├── indicators.py           # 技术指标
│   ├── metrics.py              # 风险收益指标
│   ├── backtest.py             # 信号生成、回测、基准对比
│   ├── universe.py             # 多标的批量分析与行业聚合
│   ├── report.py               # Plotly 图表与 HTML 报告
│   ├── demo.py                 # 离线合成数据
│   └── cli.py                  # 子命令解析与输出
├── examples/synthetic_demo.py
├── tests/                      # pytest 单元测试
└── output/                     # 报告与 CSV 输出
```

## 快速开始

```powershell
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 配置 Tushare token（不要把 token 写进代码或提交到仓库）
$env:TUSHARE_TOKEN="你的 token"

# 3. 不联网先跑一遍完整流程（合成数据）
python main.py demo

# 4. 单只股票技术分析
python main.py analyze --code 000338.SZ --start 20200101 --end 20231231

# 5. 双均线策略回测，对比沪深 300
python main.py backtest --code 000338.SZ --strategy ma --fast 5 --slow 20

# 6. RSI 均值回归
python main.py backtest --code 000338.SZ --strategy rsi --entry 30 --exit-level 70

# 7. 按行业批量筛选（示例：银行，最多 20 只）
python main.py screen --industry 银行 --limit 20 --start 20220101 --end 20231231
```

运行后默认在 `output/` 下生成：

| 命令 | 输出 |
| --- | --- |
| `analyze` | `000338.SZ_analysis.html` |
| `backtest` | `000338.SZ_backtest.html` |
| `screen` | `screen_银行.csv` |
| `demo` | `demo_report.html` |

## 关键设计

### 1. 增量缓存而不是每次全量拉取

日线表以 `(ts_code, trade_date)` 为主键。每次请求先查询本地已缓存的日期范围，
只补拉缺失的前后区间；`--refresh` 时才强制重拉指定区间。
这样反复调试策略时不会反复消耗 Tushare 积分，也显著缩短运行时间。

### 2. 明确避免未来函数

信号在 `t` 日收盘后生成，回测中将持仓整体 `shift(1)`，即 `t+1` 日才计入收益。
单元测试 `test_no_lookahead_bias` 固定验证了这一点。

### 3. 把交易成本算进去

换手率 × (手续费 + 滑点) 直接从策略收益中扣除。默认双边手续费 0.03%、滑点 0.02%，
可通过环境变量调整。不计算成本的回测结果通常会被高估，尤其是高换手策略。

### 4. 指标口径统一

所有绩效指标以日收益率序列为输入，年化统一按 252 个交易日：
- 年化收益：`(1 + r).prod() ** (252 / n) - 1`
- 夏普比率：日超额收益均值 / 标准差 × √252，无风险利率默认 2%
- 最大回撤：净值 / 历史最高净值 - 1 的最小值

### 5. 测试覆盖核心计算

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖：均线取值、RSI 边界与单调行情、布林带顺序、MACD 柱定义、
最大回撤已知路径、回测无未来函数、交易成本影响、基准对齐。

## 已知局限

- 日频、单标的、收盘价成交的简化回测，未考虑涨跌停、停牌、分红再投资与冲击成本；
- 策略仅用于验证流程，参数未做样本外优化，不构成投资建议；
- 财务指标接口已封装，但尚未纳入选股因子；
- Tushare 部分接口需要相应积分权限，数据可用性取决于账号权限。
