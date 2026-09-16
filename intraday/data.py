"""Data helpers for the same-day intraday spillover hypothesis.

Hypothesis: when a leader stock FIRST touches its daily price-limit (涨停) at some
specific intraday bar, a follower's price is more likely to rise over the following
`lag_bars` bars, WITHIN THE SAME TRADING DAY, than its own baseline propensity.

This reuses `limitup.data.limit_pct_for_code` for the board-based limit-percentage
rule (see that module's docstring for the ST/new-listing caveats, which apply
identically here) and `leadlag.factor`'s statistical core via the "same-row-aligned"
entry points (`compute_pairwise_stats_same_row`, `validate_out_of_sample_same_row`) -
because the follower's outcome here is already a forward-looking quantity computed at
the trigger bar itself, not something needing an external row-shift the way leadlag's
daily "next day" comparison does.

IMPORTANT CAVEAT ON THE MINUTE-BAR INDEX FORMAT: `leadlag.data.panel_from_field_dict`
(reused here) parses `get_market_data_ex`'s returned index via `pd.to_datetime`, which
has only been exercised against real REAL daily-bar output ("YYYYMMDD"-style strings)
in this project so far. Minute bars are commonly "YYYYMMDDHHMMSS"-style strings, which
pandas' datetime parser also auto-detects, but this has NOT been verified against a
real xtdata minute-bar response - if `fetch_intraday_panels_xtdata` raises a parsing
error or produces an index that doesn't look right, check the raw dict xtdata.
get_market_data_ex returns before assuming the rest of this module is wrong.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from leadlag.data import panel_from_field_dict
from limitup.data import limit_pct_for_code


def fetch_intraday_panels_xtdata(
    stock_list: list[str], start_time: str = "", end_time: str = "", period: str = "5m",
    download: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch minute-bar close + suspend-flag panels, plus the DAILY close series needed
    to compute each day's limit price (fixed for the whole day, based on the PRIOR
    trading day's close - not something you can derive from intraday bars alone).

    Minute-bar data volume is much larger than daily: a full trading day is ~240
    one-minute bars (48 five-minute bars). Scope `stock_list` and the date range down
    (few hundred liquid/active names, months rather than years) before running this
    against the whole market - see intraday/README.md.

    Returns (minute_close, minute_suspend_flag, daily_close), all UNADJUSTED
    (dividend_type='none') since limit-price arithmetic needs raw price levels (see
    limitup.data's fetch function for the same reasoning).
    """
    from xtquant import xtdata

    if download:
        for i, code in enumerate(stock_list, 1):
            xtdata.download_history_data(code, period, start_time, end_time)
            if i % 100 == 0 or i == len(stock_list):
                print(f"[xtdata] downloaded {i}/{len(stock_list)}")

    minute_raw = xtdata.get_market_data_ex(
        ["close", "suspendFlag"], stock_list, period=period,
        start_time=start_time, end_time=end_time, dividend_type="none", fill_data=False,
    )
    minute_close = panel_from_field_dict(minute_raw, "close")
    minute_suspend = panel_from_field_dict(minute_raw, "suspendFlag")

    daily_raw = xtdata.get_market_data_ex(
        ["close"], stock_list, period="1d",
        start_time=start_time, end_time=end_time, dividend_type="none", fill_data=False,
    )
    daily_close = panel_from_field_dict(daily_raw, "close")
    return minute_close, minute_suspend, daily_close


def broadcast_prev_close_to_bars(minute_index: pd.DatetimeIndex, daily_close: pd.DataFrame) -> pd.DataFrame:
    """For every intraday bar, look up the PREVIOUS trading day's close (that whole
    day's limit-price basis) and broadcast it across every bar of that day.
    """
    daily_prev_close = daily_close.sort_index().shift(1)
    bar_dates = minute_index.normalize()
    broadcasted = daily_prev_close.reindex(bar_dates)
    broadcasted.index = minute_index
    return broadcasted


def compute_first_touch_indicator(
    minute_close: pd.DataFrame, prev_close_by_bar: pd.DataFrame, valid: pd.DataFrame, tolerance: float = 0.003,
) -> pd.DataFrame:
    """Boolean panel: is this the FIRST bar, on its trading day, where the stock's
    price reached its daily limit? (Not every bar it stays sealed - exactly one event
    per stock per day at most, keeping trigger observations roughly independent for
    the z-test's sake, the same way limitup's daily version is a once-per-day event.)
    """
    limit_pct = pd.Series({c: limit_pct_for_code(c) for c in minute_close.columns})
    limit_price = prev_close_by_bar.mul(1 + limit_pct, axis=1).round(2)
    touched = (minute_close >= limit_price - tolerance) & valid

    dates = minute_close.index.normalize()
    cum_touched = touched.groupby(dates).cumsum()
    first_touch = touched & (cum_touched == 1)
    return first_touch.fillna(False)


def compute_forward_outcome(
    minute_close: pd.DataFrame, valid_raw: pd.DataFrame, lag_bars: int,
    mode: str = "absolute", threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every bar t, did the price rise from t to t + `lag_bars`, WITHOUT that
    window crossing into the next trading day? Indexed by the STARTING bar t (unlike
    leadlag's daily "up on day t+lag" indexed by the later day) - see
    leadlag.factor.compute_pairwise_stats_same_row's docstring for why this matters.

    mode='excess' nets out the bar's cross-sectional median forward return first, same
    rationale as leadlag.factor.compute_up_indicator (don't mistake a common intraday
    market move for a leader-follower relationship).
    """
    if lag_bars <= 0 or lag_bars >= len(minute_close):
        raise ValueError("lag_bars must be a positive number smaller than the number of bars")

    idx = minute_close.index
    codes = minute_close.columns
    dates = idx.normalize().to_numpy()
    close = minute_close.to_numpy(dtype=np.float64)
    valid_arr = valid_raw.to_numpy(dtype=bool)
    n = len(idx)

    shifted_close = np.full_like(close, np.nan)
    shifted_close[:-lag_bars] = close[lag_bars:]
    shifted_valid = np.zeros_like(valid_arr)
    shifted_valid[:-lag_bars] = valid_arr[lag_bars:]
    same_day = np.zeros(n, dtype=bool)
    same_day[:-lag_bars] = dates[lag_bars:] == dates[:-lag_bars]

    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_ret = shifted_close / close - 1

    if mode == "excess":
        # the last `lag_bars` rows are all-NaN by construction (no forward window left)
        # - np.nanmedian warns on an all-NaN slice even though the NaN it returns is
        # exactly right and gets masked out by `valid` below regardless.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            market = np.nanmedian(fwd_ret, axis=1, keepdims=True)
        signal = fwd_ret - market
    elif mode == "absolute":
        signal = fwd_ret
    else:
        raise ValueError(f"unknown mode: {mode!r}, expected 'absolute' or 'excess'")

    valid = valid_arr & shifted_valid & same_day[:, None]
    up = (signal > threshold) & valid

    return pd.DataFrame(up, index=idx, columns=codes), pd.DataFrame(valid, index=idx, columns=codes)


def make_synthetic_intraday_market(
    n_days: int = 500,
    bars_per_day: int = 48,
    n_stocks: int = 40,
    n_pairs: int = 6,
    lag_bars: int = 6,
    limitup_prob: float = 0.1,
    flip_prob: float = 0.6,
    boost: float = 0.025,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """Fabricate an intraday market with injected 'leader first-touches-limit at bar t
    -> follower up by bar t + lag_bars, same day' relationships.

    `bars_per_day=48` matches a real A-share trading day's 4 hours at 5-minute bars.
    Prices compound with simple returns, rounded to 2 decimals every bar (same
    technique as `limitup.data.make_synthetic_limitup_market`, for the same reason:
    forcing a leader's return to its exact `limit_pct_for_code` produces a close that
    matches `compute_first_touch_indicator`'s own formula precisely).

    Returns (minute_close, minute_suspend, daily_close, injected_pairs).
    """
    rng = np.random.default_rng(seed)
    T = n_days * bars_per_day
    codes = [f"300{i:03d}.SZ" if i % 5 == 0 else f"600{i:03d}.SH" for i in range(n_stocks)]
    limit_pct = np.array([limit_pct_for_code(c) for c in codes])

    market = rng.normal(0, 0.001, T)
    idio = rng.normal(0, 0.0015, (T, n_stocks))
    rets = market[:, None] + idio

    shuffled = codes.copy()
    rng.shuffle(shuffled)
    pairs = [(shuffled[2 * k], shuffled[2 * k + 1]) for k in range(n_pairs)]

    # trigger_bar[d, i] = the bar-of-day at which stock i is forced to touch its limit
    # on day d, or -1. Recorded here (return-space additions for the follower boost can
    # be applied right away) but APPLIED as an absolute price override below, once the
    # day-by-day simulation actually knows that day's opening basis (see the note there
    # on why a %-return relative to the previous BAR is the wrong thing to force).
    trigger_bar = -np.ones((n_days, n_stocks), dtype=int)
    max_start = bars_per_day - lag_bars - 1

    for leader, follower in pairs:
        li, fi = codes.index(leader), codes.index(follower)
        if max_start <= 0:
            continue
        trigger_days = np.where(rng.random(n_days) < limitup_prob)[0]
        for d in trigger_days:
            bar_in_day = rng.integers(0, max_start + 1)
            trigger_bar[d, li] = bar_in_day
            t = d * bars_per_day + bar_in_day

            if rng.random() < flip_prob:
                boost_t = t + lag_bars
                rets[boost_t, fi] += boost
                # de-mean WITHIN THE SAME DAY only, so this doesn't leak a drift into
                # other days (see limitup.data's synthetic generator for the same
                # rationale at daily granularity).
                day_start, day_end = d * bars_per_day, (d + 1) * bars_per_day
                non_boosted = np.ones(bars_per_day, dtype=bool)
                non_boosted[boost_t - day_start] = False
                day_rows = np.arange(day_start, day_end)[non_boosted]
                rets[day_rows, fi] -= boost / non_boosted.sum()

    # Day-by-day price simulation so a leader's trigger bar can be pinned to an
    # ABSOLUTE price level derived from THAT DAY's actual opening basis (the previous
    # day's real closing price). Forcing rets[t] = limit_pct relative to the
    # immediately preceding INTRADAY bar (the first version of this function did that)
    # overshoots the true limit price whenever the stock already drifted intraday
    # before the trigger bar - exactly the bug this loop exists to avoid.
    price = np.empty((T, n_stocks))
    day_basis = np.full(n_stocks, 10.0)
    for d in range(n_days):
        day_start = d * bars_per_day
        prev_bar = day_basis.copy()
        triggers_today = trigger_bar[d]
        for b in range(bars_per_day):
            t = day_start + b
            bar_price = prev_bar * (1 + rets[t])
            hit = triggers_today == b
            if hit.any():
                bar_price[hit] = day_basis[hit] * (1 + limit_pct[hit])
            bar_price = np.round(bar_price, 2)
            price[t] = bar_price
            prev_bar = bar_price
        day_basis = prev_bar

    business_days = pd.bdate_range("2022-01-01", periods=n_days)
    minute_offsets = pd.timedelta_range("0min", periods=bars_per_day, freq="5min")
    timestamps = [
        pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30) + off
        for day in business_days for off in minute_offsets
    ]
    index = pd.DatetimeIndex(timestamps)

    minute_close = pd.DataFrame(price, index=index, columns=codes)
    minute_suspend = pd.DataFrame(0, index=index, columns=codes)
    daily_close = minute_close.groupby(minute_close.index.normalize()).last()
    return minute_close, minute_suspend, daily_close, pairs
