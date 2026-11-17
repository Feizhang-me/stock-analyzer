"""Tushare 数据访问层。

设计目标：
- 统一封装股票列表、日线、指数、财务指标接口；
- 使用 SQLite 做本地缓存，日线按区间增量更新，避免重复请求；
- 简单限流 + 失败重试，降低批量拉取时的报错率；
- API 不可用时自动回退到本地缓存。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DAILY_API_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
)
NUMERIC_COLUMNS = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
STOCK_BASIC_FIELDS = "ts_code,symbol,name,area,industry,market,list_date"


class DataSourceError(RuntimeError):
    """数据源不可用或返回异常。"""


TUSHARE_HOSTS = ("api.tushare.pro", "api.waditu.com")


def _bypass_system_proxy_for_tushare() -> None:
    """让 Tushare 直连，避免系统代理返回 502 时被误判成“没有数据”。

    Windows 上 requests 会读取 IE/系统代理（如 Clash 的 127.0.0.1:7897）。
    代理一旦拦截 api.tushare.pro，tushare 客户端只会返回空表而不报错，
    这里把 Tushare 的主机名追加进 no_proxy，保持其它请求不动。
    """
    raw = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    hosts = [part.strip() for part in raw.split(",") if part.strip()]
    missing = [host for host in TUSHARE_HOSTS if host not in hosts]
    if not missing:
        return
    merged = ",".join(hosts + missing)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _fmt_date(value: str | date | datetime) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y%m%d")
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"非法日期: {value!r}，应为 YYYYMMDD 或 YYYY-MM-DD")
    return text


def _shift_day(value: str, days: int) -> str:
    return (datetime.strptime(value, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


class TushareDataSource:
    """带 SQLite 缓存的 Tushare Pro 客户端。"""

    def __init__(
        self,
        token: str,
        cache_path: str | Path,
        *,
        requests_per_minute: int = 200,
        max_retries: int = 3,
        retry_wait: float = 1.0,
    ) -> None:
        if not token:
            raise DataSourceError("缺少 Tushare token，请设置环境变量 TUSHARE_TOKEN")
        _bypass_system_proxy_for_tushare()
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
            raise DataSourceError("未安装 tushare，请先执行 pip install -r requirements.txt") from exc

        ts.set_token(token)
        self._pro = ts.pro_api()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._last_call = 0.0
        self.max_retries = max(1, max_retries)
        self.retry_wait = retry_wait
        self._init_db()

    # ---------- 基础设施 ----------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cache_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_basic (
                    ts_code TEXT PRIMARY KEY, symbol TEXT, name TEXT, area TEXT,
                    industry TEXT, market TEXT, list_date TEXT, updated_at TEXT
                )
                """
            )
            for table in ("daily", "index_daily"):
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        ts_code TEXT, trade_date TEXT, open REAL, high REAL, low REAL,
                        close REAL, pre_close REAL, change REAL, pct_chg REAL,
                        vol REAL, amount REAL,
                        PRIMARY KEY (ts_code, trade_date)
                    )
                    """
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_cache (
                    cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _call(self, func, **kwargs) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                df = func(**kwargs)
                return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            except Exception as exc:  # Tushare 的异常类型不统一
                last_error = exc
                if attempt < self.max_retries:
                    wait = self.retry_wait * (2 ** (attempt - 1))
                    logger.warning("Tushare 调用失败（第 %s 次），%.1fs 后重试: %s", attempt, wait, exc)
                    time.sleep(wait)
        raise DataSourceError(f"Tushare 调用失败: {last_error}") from last_error

    # ---------- 股票列表 ----------

    def stock_basic(self, refresh: bool = False, max_age_days: int = 7) -> pd.DataFrame:
        with self._connect() as conn:
            cached = pd.read_sql_query(
                "SELECT ts_code, symbol, name, area, industry, market, list_date FROM stock_basic",
                conn,
            )
            meta = conn.execute(
                "SELECT value FROM meta WHERE key = 'stock_basic_updated_at'"
            ).fetchone()

        fresh = False
        if meta:
            try:
                updated = datetime.fromisoformat(meta[0])
                fresh = (datetime.now() - updated) <= timedelta(days=max_age_days)
            except ValueError:
                fresh = False

        if not refresh and fresh and not cached.empty:
            return cached

        try:
            df = self._call(
                self._pro.stock_basic,
                exchange="",
                list_status="L",
                fields=STOCK_BASIC_FIELDS,
            )
        except DataSourceError:
            if not cached.empty:
                logger.warning("股票列表接口不可用，使用本地缓存")
                return cached
            raise

        if df.empty:
            return cached

        df = df.copy()
        df["list_date"] = df["list_date"].astype(str)
        df["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("DELETE FROM stock_basic")
            df.to_sql("stock_basic", conn, if_exists="append", index=False)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('stock_basic_updated_at', ?)",
                (datetime.now().isoformat(timespec="seconds"),),
            )
        return df.drop(columns=["updated_at"])

    # ---------- 行情：增量缓存 ----------

    def _load_incremental(
        self,
        table: str,
        api_func,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool,
    ) -> pd.DataFrame:
        start = _fmt_date(start_date)
        end = _fmt_date(end_date)
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")

        if refresh:
            with self._connect() as conn:
                conn.execute(
                    f"DELETE FROM {table} WHERE ts_code = ? AND trade_date BETWEEN ? AND ?",
                    (ts_code, start, end),
                )

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT MIN(trade_date), MAX(trade_date) FROM {table} WHERE ts_code = ?",
                (ts_code,),
            ).fetchone()

        min_cached, max_cached = row if row else (None, None)
        windows: list[tuple[str, str]] = []
        if not min_cached or not max_cached:
            windows.append((start, end))
        else:
            if start < min_cached:
                windows.append((start, _shift_day(min_cached, -1)))
            if end > max_cached:
                windows.append((_shift_day(max_cached, 1), end))

        errors: list[str] = []
        for win_start, win_end in windows:
            if win_start > win_end:
                continue
            try:
                df = self._call(
                    api_func,
                    ts_code=ts_code,
                    start_date=win_start,
                    end_date=win_end,
                    fields=DAILY_API_FIELDS,
                )
            except DataSourceError as exc:
                # 接口报错（例如积分不足、代理异常）时先记录下来，
                # 若有本地缓存仍可继续；否则下面会带着真实原因报错。
                errors.append(str(exc))
                logger.warning("拉取 %s %s-%s 失败: %s", ts_code, win_start, win_end, exc)
                continue
            if df.empty:
                continue
            df = self._normalize_daily(df)
            store = df.reset_index()
            store["trade_date"] = pd.to_datetime(store["trade_date"]).dt.strftime("%Y%m%d")
            with self._connect() as conn:
                store.to_sql("_incoming", conn, if_exists="replace", index=False)
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {table}
                    (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                    SELECT ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
                    FROM _incoming
                    """
                )
                conn.execute("DROP TABLE _incoming")

        with self._connect() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
                FROM {table}
                WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date
                """,
                conn,
                params=(ts_code, start, end),
            )
        result = self._normalize_daily(df, already_sorted=True)
        if result.empty and errors:
            hint = ""
            if any("权限" in message or "40203" in message for message in errors):
                hint = (
                    "\n提示：Tushare 返回 40203 表示当前账号积分不足。"
                    "daily 需要 120 积分起，index_daily 需要 2000 积分起，"
                    "请登录 https://tushare.pro 查看积分与接口权限。"
                )
            elif any("token" in message or "40101" in message for message in errors):
                hint = (
                    "\n提示：Tushare 返回 40101 表示这个 token 已失效（或填错了）。"
                    "请登录 https://tushare.pro 个人中心重新复制“接口TOKEN”，"
                    "更新 .env 里的 TUSHARE_TOKEN 后再运行。"
                )
            raise DataSourceError(f"获取 {ts_code} 行情失败：{errors[0]}{hint}")
        return result

    @staticmethod
    def _normalize_daily(df: pd.DataFrame, already_sorted: bool = False) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["ts_code", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
            )
        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
        out = out.dropna(subset=["trade_date"])
        if not already_sorted:
            out = out.sort_values("trade_date")
        out = out.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        for col in NUMERIC_COLUMNS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.set_index("trade_date")
        out.index.name = "trade_date"
        return out

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        return self._load_incremental(
            "daily", self._pro.daily, ts_code, start_date, end_date, refresh
        )

    def index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        return self._load_incremental(
            "index_daily", self._pro.index_daily, index_code, start_date, end_date, refresh
        )

    # ---------- 财务指标 ----------

    def fina_indicator(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
        max_age_days: int = 30,
    ) -> pd.DataFrame:
        start, end = _fmt_date(start_date), _fmt_date(end_date)
        cache_key = f"{ts_code}_{start}_{end}"

        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, updated_at FROM financial_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if row and not refresh:
            updated = datetime.fromisoformat(row[1])
            if (datetime.now() - updated) <= timedelta(days=max_age_days):
                return pd.DataFrame(json.loads(row[0]))

        try:
            df = self._call(
                self._pro.fina_indicator,
                ts_code=ts_code,
                start_date=start,
                end_date=end,
            )
        except DataSourceError:
            if row:
                logger.warning("财务接口不可用，使用过期缓存: %s", ts_code)
                return pd.DataFrame(json.loads(row[0]))
            raise

        if not df.empty:
            payload = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, default=str)
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO financial_cache(cache_key, payload, updated_at) VALUES (?, ?, ?)",
                    (cache_key, payload, datetime.now().isoformat(timespec="seconds")),
                )
        return df

    # ---------- 便捷方法 ----------

    def industry_stocks(self, industry: str) -> list[str]:
        basic = self.stock_basic()
        return basic.loc[basic["industry"] == industry, "ts_code"].tolist()

    def stock_name(self, ts_code: str) -> str:
        basic = self.stock_basic()
        match = basic.loc[basic["ts_code"] == ts_code, "name"]
        return str(match.iloc[0]) if not match.empty else ts_code
