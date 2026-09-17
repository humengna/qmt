"""Data helpers specific to the limit-up-trigger hypothesis.

The "leader" event here is much more specific than leadlag's plain "up": a stock
closing AT its daily price-limit (涨停), not just closing higher. That event is
derivable from ordinary daily bars (close vs. a computed limit price) - no minute/tick
data needed. Everything else (same-sector restriction, FDR-controlled mining,
out-of-sample validation, backtest engine) is reused from `leadlag` via the asymmetric
functions in `leadlag.factor` (the leader's trigger and the follower's outcome are
different signals here, unlike leadlag's own symmetric use of "up" on both sides).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.data import panel_from_field_dict


def limit_pct_for_code(code: str) -> float:
    """Board-based daily price-limit percentage, inferred from the stock code prefix.

    Simplification: ignores the ST/*ST 5% limit (use `build_st_exclusion_set_xtdata`
    instead of trying to reconstruct historical ST status) and the no-limit first
    trading day for new listings. Both only ever make the computed limit price too
    HIGH for the affected stock-days, so the failure mode is a missed true limit-up
    (false negative), never a fabricated one (false positive) - the safer direction to
    be wrong in for a mining pipeline whose main risk is spurious relationships.
    """
    stock = code.split(".")[0]
    if stock.startswith(("300", "301", "688")):
        return 0.20
    if stock.startswith(("8", "4", "92")):
        return 0.30  # 北交所, approximate
    return 0.10


def compute_limit_up_indicator(
    close: pd.DataFrame, preclose: pd.DataFrame, valid: pd.DataFrame, tolerance: float = 0.003,
) -> pd.DataFrame:
    """Boolean panel: did the stock close AT (or within `tolerance` of) its limit that day?

    The limit price is `round(preclose * (1 + limit_pct_for_code), 2)`, matching
    exchange convention (2 decimal places); `tolerance` absorbs residual rounding edge
    cases against the exchange's own computed value. Requires UNADJUSTED prices
    (`dividend_type='none'` when fetching) - the ratio comparison isn't reliable
    against post-adjustment price levels.
    """
    limit_pct = pd.Series({c: limit_pct_for_code(c) for c in close.columns})
    limit_price = preclose.mul(1 + limit_pct, axis=1).round(2)
    hit = (close >= limit_price - tolerance) & valid
    return hit.fillna(False)


def build_st_exclusion_set_xtdata(stock_list: list[str]) -> set[str]:
    """Best-effort set of CURRENTLY ST/*ST-flagged codes to exclude from mining.

    Uses each stock's CURRENT instrument name only - it does not reconstruct
    historical ST status, so a name that was ST in the past but isn't now (or vice
    versa) is classified by today's status for its ENTIRE history. Coarse but safe:
    serious strategies avoid ST names anyway given delisting/manipulation risk, and a
    slightly wrong boundary here doesn't invalidate anything downstream. Fails soft
    (prints a warning, returns an empty set) if the detail lookup isn't available in
    your xtquant version, rather than blocking the whole run.
    """
    from xtquant import xtdata

    getter = getattr(xtdata, "get_instrument_detail", None) or getattr(xtdata, "get_instrumentdetail", None)
    if getter is None:
        print("[limitup] no get_instrument_detail-like function found in xtdata - skipping ST filtering")
        return set()

    st_codes: set[str] = set()
    for code in stock_list:
        try:
            detail = getter(code)
        except Exception:  # noqa: BLE001 - best-effort; one bad code must not abort the run
            continue
        name = (detail or {}).get("InstrumentName", "") if isinstance(detail, dict) else ""
        if "ST" in name.upper():
            st_codes.add(code)
    return st_codes


def fetch_limitup_panels_xtdata(
    stock_list: list[str], start_time: str = "", end_time: str = "", period: str = "1d",
    download: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch what limitup/event.py needs: raw prices for exact limit-price arithmetic,
    plus back-adjusted close for the follower-side return/up computation (consistent
    with leadlag's own convention). Two separate `get_market_data_ex` calls since
    `dividend_type` can't be mixed within one.

    Returns (raw_close, raw_preclose, suspend_flag, adjusted_close).
    """
    from xtquant import xtdata

    if download:
        for i, code in enumerate(stock_list, 1):
            xtdata.download_history_data(code, period, start_time, end_time)
            if i % 200 == 0 or i == len(stock_list):
                print(f"[xtdata] downloaded {i}/{len(stock_list)}")

    raw = xtdata.get_market_data_ex(
        ["close", "preClose", "suspendFlag"], stock_list, period=period,
        start_time=start_time, end_time=end_time, dividend_type="none", fill_data=False,
    )
    raw_close = panel_from_field_dict(raw, "close")
    raw_preclose = panel_from_field_dict(raw, "preClose")
    suspend = panel_from_field_dict(raw, "suspendFlag")

    adj = xtdata.get_market_data_ex(
        ["close"], stock_list, period=period, start_time=start_time, end_time=end_time,
        dividend_type="back_ratio", fill_data=False,
    )
    adj_close = panel_from_field_dict(adj, "close")
    return raw_close, raw_preclose, suspend, adj_close


def make_synthetic_limitup_market(
    n_stocks: int = 40,
    n_days: int = 800,
    n_pairs: int = 5,
    lag: int = 1,
    limitup_prob: float = 0.05,
    flip_prob: float = 0.5,
    boost: float = 0.03,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """Fabricate a market with injected 'leader hits limit-up -> follower up' relationships.

    Prices compound with simple (not log) returns and are rounded to 2 decimals at
    every step, exactly like real exchange quotes, so forcing a leader's return to its
    `limit_pct_for_code` on a trigger day produces a close that EXACTLY matches
    `compute_limit_up_indicator`'s own formula - a faithful test of the detector, not
    an approximation. Codes are split across the 20%-limit board (300xxx) and the
    10%-limit board (600xxx) so `limit_pct_for_code` itself gets exercised on both.

    Returns (raw_close, raw_preclose, suspend, adj_close, injected_pairs). `adj_close`
    is identical to `raw_close` here (no dividends are simulated).
    """
    rng = np.random.default_rng(seed)
    codes = [f"300{i:03d}.SZ" if i % 5 == 0 else f"600{i:03d}.SH" for i in range(n_stocks)]
    limit_pct = np.array([limit_pct_for_code(c) for c in codes])

    market = rng.normal(0, 0.01, n_days)
    idio = rng.normal(0, 0.015, (n_days, n_stocks))
    rets = market[:, None] + idio

    shuffled = codes.copy()
    rng.shuffle(shuffled)
    pairs = [(shuffled[2 * k], shuffled[2 * k + 1]) for k in range(n_pairs)]

    for leader, follower in pairs:
        li, fi = codes.index(leader), codes.index(follower)
        trigger_days = rng.random(n_days) < limitup_prob
        trigger_days[n_days - lag:] = False  # leave room for the lagged follower day
        rets[trigger_days, li] = limit_pct[li]  # force an EXACT limit-up close that day

        trig_idx = np.where(trigger_days)[0]
        flip = rng.random(len(trig_idx)) < flip_prob
        boosted_days = trig_idx[flip] + lag
        rets[boosted_days, fi] += boost
        # de-mean so the follower's price path doesn't drift just from being a follower
        # (see leadlag.data.make_synthetic_market for the same technique/rationale)
        non_boosted = np.ones(n_days, dtype=bool)
        non_boosted[boosted_days] = False
        if len(boosted_days) > 0 and non_boosted.sum() > 0:
            drag = boost * len(boosted_days) / non_boosted.sum()
            rets[non_boosted, fi] -= drag

    price = np.empty((n_days, n_stocks))
    prev = np.full(n_stocks, 10.0)
    for t in range(n_days):
        price[t] = np.round(prev * (1 + rets[t]), 2)
        prev = price[t]
    preclose = np.vstack([np.full((1, n_stocks), 10.0), price[:-1]])

    dates = pd.bdate_range("2020-01-01", periods=n_days)
    raw_close = pd.DataFrame(price, index=dates, columns=codes)
    raw_preclose = pd.DataFrame(preclose, index=dates, columns=codes)
    suspend = pd.DataFrame(0, index=dates, columns=codes)
    adj_close = raw_close.copy()
    return raw_close, raw_preclose, suspend, adj_close, pairs
