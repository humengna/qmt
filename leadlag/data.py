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
