"""Data loading adapters.

`leadlag.factor` and `leadlag.backtest` only ever deal in plain wide pandas
DataFrames (DatetimeIndex x stock-code columns). This module is the only place
that knows about a specific data source, so the rest of the package can be unit
tested without a live QMT connection.

Sources supported:
  - xtquant.xtdata: the standalone QMT/MiniQMT SDK, usable from a normal Python
    process (terminal, venv, Jupyter) as long as a QMT/MiniQMT terminal is
    running locally. This is the recommended source for whole-market research
    since it is not subject to the strategy editor's 500-symbol live-subscribe
    cap (see docs/QMT_API_NOTES.md).
  - CSV: wide-format files you already have (date index, one column per stock).
  - synthetic: a fabricated multi-stock market with a handful of known, injected
    lead-lag relationships. Used for tests and for smoke-testing the pipeline
    without any market data at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def panel_from_field_dict(data: dict, field: str) -> pd.DataFrame:
    """Reshape `{stock_code: DataFrame(index=time, columns=fields)}` into one wide panel.

    This is exactly the return shape documented for `ContextInfo.get_market_data_ex`
    (and mirrored by the standalone `xtquant.xtdata.get_market_data_ex`).
    """
    series = {code: df[field] for code, df in data.items() if field in df.columns and not df.empty}
    panel = pd.DataFrame(series)
    if not panel.empty:
        panel.index = pd.to_datetime(panel.index.astype(str))
    return panel.sort_index()


def fetch_price_panels_xtdata(
    stock_list: list[str],
    start_time: str = "",
    end_time: str = "",
    period: str = "1d",
    dividend_type: str = "back_ratio",
    download: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (open, close) panels for a universe via the standalone `xtquant.xtdata` SDK.

    Requires a running QMT/MiniQMT terminal with local market data. Run this from a
    normal Python process, not from inside the strategy-editor sandbox.
    """
    from xtquant import xtdata  # lazy import: only needed when actually pulling from QMT

    if download:
        for i, code in enumerate(stock_list, 1):
            xtdata.download_history_data(code, period, start_time, end_time)
            if i % 200 == 0 or i == len(stock_list):
                print(f"[xtdata] downloaded {i}/{len(stock_list)}")

    raw = xtdata.get_market_data_ex(
        ["open", "close", "suspendFlag"],
        stock_list,
        period=period,
        start_time=start_time,
        end_time=end_time,
        dividend_type=dividend_type,
        fill_data=False,
    )
    open_panel = panel_from_field_dict(raw, "open")
    close_panel = panel_from_field_dict(raw, "close")
    return open_panel, close_panel


def get_full_market_stock_list_xtdata(sectors: tuple[str, ...] = ("沪深A股",)) -> list[str]:
    from xtquant import xtdata

    codes: list[str] = []
    for sector in sectors:
        codes.extend(xtdata.get_stock_list_in_sector(sector))
    return sorted(set(codes))


def rank_stocks_by_liquidity_xtdata(
    stock_list: list[str],
    start_time: str = "",
    end_time: str = "",
    period: str = "1d",
    top_n: int | None = None,
) -> list[str]:
    """Rank a universe by median daily traded value (成交额) and optionally keep the top_n.

    Mining the whole market pays for it twice: O(N^2) compute, and a much stricter
    FDR bar (the correction gets harsher the more pairs you test at once - see
    README's "统计陷阱说明"). Shrinking to the most liquid names first cuts both, and
    also drops names you likely couldn't fill an order in anyway. Reads 'amount' from
    the already-downloaded local cache, so it doesn't re-trigger any history download.
    """
    from xtquant import xtdata

    raw = xtdata.get_market_data_ex(
        ["amount"], stock_list, period=period, start_time=start_time, end_time=end_time,
        fill_data=False,
    )
    amount = panel_from_field_dict(raw, "amount")
    ranked = amount.median(axis=0).sort_values(ascending=False).index.tolist()
    return ranked[:top_n] if top_n else ranked


# Best-effort default for --sector-names: 申万一级行业 (2021 revision) names, which many
# QMT/xtdata data vendors mirror as sector-tree entries. THIS IS A GUESS, not something
# verified against your specific broker/data vendor's actual sector list - names can
# differ (some vendors use 中信行业分类 instead, or prefix/suffix the name differently).
# build_sector_map_xtdata() prints per-name coverage precisely so a mismatch is obvious
# rather than silently dropping stocks; use list_sectors_xtdata() to find the real names
# in your client if the default's coverage looks low.
DEFAULT_SW_L1_SECTORS = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器", "食品饮料",
    "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输", "房地产", "商贸零售",
    "社会服务", "银行", "非银金融", "综合", "建筑材料", "建筑装饰", "电力设备",
    "国防军工", "计算机", "传媒", "通信", "机械设备", "煤炭", "石油石化", "环保", "美容护理",
]


def list_sectors_xtdata(node: str = "") -> tuple[list[str], list[str]]:
    """List the QMT client's sector-tree entries under `node` ('' = top level).

    Thin wrapper over `xtdata.get_sector_list`, returned as (sector_names, folder_names).
    A folder name can itself be passed back in as `node` to look one level deeper. Use
    this to find the exact industry/concept board names your client actually has before
    relying on DEFAULT_SW_L1_SECTORS or passing your own --sector-names.
    """
    from xtquant import xtdata

    info = xtdata.get_sector_list(node)
    sector_names, folder_names = (info[0], info[1]) if info else ([], [])
    return list(sector_names), list(folder_names)


def build_sector_map_xtdata(sector_names: list[str]) -> dict[str, str]:
    """Map each stock to the first `sector_names` entry it belongs to.

    Built from `xtdata.get_stock_list_in_sector`, the same call used for the whole-market
    universe, just pointed at named industry/concept boards instead (see
    DEFAULT_SW_L1_SECTORS). A stock in more than one of the given sectors is assigned to
    whichever is listed first - pass a mutually-exclusive classification (like one
    industry standard's L1 categories) to avoid that ambiguity. Prints per-sector counts
    so a name that doesn't exist in your client (which quietly returns 0 stocks rather
    than erroring) is obvious rather than silently shrinking your mapped universe.
    """
    from xtquant import xtdata

    mapping: dict[str, str] = {}
    for sector in sector_names:
        codes = xtdata.get_stock_list_in_sector(sector)
        new_codes = [c for c in codes if c not in mapping]
        for c in new_codes:
            mapping[c] = sector
        print(f"[sector] {sector}: {len(codes)} stocks ({len(new_codes)} newly assigned)")
    return mapping


def load_price_panels_csv(close_path: str, open_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load wide-format CSVs: a date index column plus one column per stock code."""
    close_panel = pd.read_csv(close_path, index_col=0, parse_dates=True).sort_index()
    if open_path:
        open_panel = pd.read_csv(open_path, index_col=0, parse_dates=True).sort_index()
    else:
        open_panel = close_panel.copy()
    return open_panel, close_panel


def make_synthetic_market(
    n_stocks: int = 60,
    n_days: int = 1500,
    n_lead_lag_pairs: int = 6,
    lag: int = 1,
    flip_prob: float = 0.4,
    boost: float = 0.04,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """Fabricate a market with a handful of injected, known lead-lag relationships.

    On days the leader closes up, the follower gets an extra positive return `boost`
    added `lag` days later with probability `flip_prob`, on top of pure market + noise.
    Used to check the mining/backtest pipeline actually recovers a real signal (and,
    on pure-noise data with n_lead_lag_pairs=0, that it does NOT hallucinate one).
    """
    rng = np.random.default_rng(seed)
    codes = [f"SIM{i:04d}.SZ" for i in range(n_stocks)]

    market = rng.normal(0, 0.01, n_days)
    idio = rng.normal(0, 0.015, (n_days, n_stocks))
    rets = market[:, None] + idio

    shuffled = codes.copy()
    rng.shuffle(shuffled)
    pairs = [(shuffled[2 * k], shuffled[2 * k + 1]) for k in range(n_lead_lag_pairs)]

    for leader, follower in pairs:
        li, fi = codes.index(leader), codes.index(follower)
        leader_up_days = np.where(rets[:, li] > 0)[0]
        leader_up_days = leader_up_days[leader_up_days < n_days - lag]
        flip = rng.random(len(leader_up_days)) < flip_prob
        boosted_days = leader_up_days[flip] + lag
        rets[boosted_days, fi] += boost

        # De-mean: spread an equal, opposite drag over the follower's non-boosted
        # days so its unconditional average return - and hence its price path - is
        # unaffected. Only the CONDITIONAL probability of an up day should shift;
        # without this, a one-directional boost compounds into an unbounded price
        # drift over a multi-year backtest, which is a synthetic-data artifact, not
        # something a real lead-lag effect would ever do to a stock's price level.
        non_boosted = np.ones(n_days, dtype=bool)
        non_boosted[boosted_days] = False
        if len(boosted_days) > 0 and non_boosted.sum() > 0:
            drag = boost * len(boosted_days) / non_boosted.sum()
            rets[non_boosted, fi] -= drag

    price = 10 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close_panel = pd.DataFrame(price, index=dates, columns=codes)
    open_panel = close_panel.shift(1).bfill()
    return open_panel, close_panel, pairs
