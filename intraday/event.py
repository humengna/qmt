"""Leader-side and follower-side signal construction for same-day intraday spillover.

All of the actual statistics (FDR-controlled mining, out-of-sample re-validation) are
reused directly from `leadlag.factor`'s same-row-aligned entry points
(`compute_pairwise_stats_same_row`, `validate_out_of_sample_same_row`) - this module
only builds the two boolean panels those functions need.
"""
from __future__ import annotations

import pandas as pd

from .data import (
    broadcast_prev_close_to_bars,
    compute_first_surge_indicator,
    compute_first_threshold_cross_indicator,
    compute_first_touch_indicator,
    compute_forward_outcome,
    compute_overnight_outcome,
)


def build_leader_frames(
    minute_close: pd.DataFrame, minute_suspend: pd.DataFrame, daily_close: pd.DataFrame, tolerance: float = 0.003,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (leader_triggered, leader_valid) panels: did the stock FIRST touch its
    daily price-limit at this specific bar? Requires UNADJUSTED prices (dividend_type
    ='none' when fetching - see `intraday.data.fetch_intraday_panels_xtdata`).
    """
    prev_close_by_bar = broadcast_prev_close_to_bars(minute_close.index, daily_close)
    valid = minute_close.notna() & (minute_suspend != 1) & prev_close_by_bar.notna()
    triggered = compute_first_touch_indicator(minute_close, prev_close_by_bar, valid, tolerance)
    return triggered, valid


def build_leader_frames_surge(
    minute_close: pd.DataFrame, minute_suspend: pd.DataFrame,
    threshold: float = 0.02, window_bars: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (leader_triggered, leader_valid) panels for a "急拉" trigger: did the
    stock FIRST rise `threshold` over the trailing `window_bars` bars at this bar?

    Both the size and the speed of the move are parameters, which is what separates this
    from `build_leader_frames` (size fixed by the board's price-limit rule) and from
    `build_leader_frames_threshold` (anchored on the day's open, so a slow grind up
    counts as much as a spike). Needs only intraday close prices - no daily_close.
    """
    valid = minute_close.notna() & (minute_suspend != 1)
    triggered = compute_first_surge_indicator(minute_close, valid, threshold, window_bars)
    return triggered, valid


def build_leader_frames_threshold(
    minute_close: pd.DataFrame, minute_suspend: pd.DataFrame, threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (leader_triggered, leader_valid) panels using a plain intraday-rise
    threshold instead of a price-limit formula: did the stock FIRST reach a cumulative
    return of `threshold` since ITS OWN DAY'S FIRST BAR, at this specific bar?

    Use this instead of `build_leader_frames` when the leader universe isn't subject to
    (or doesn't usefully hit) a 涨停/price-limit rule at all - e.g. T0-eligible
    cross-border/commodity/bond ETFs, which rarely if ever seal at a board limit. Needs
    only intraday close prices, no daily_close/limit-percentage input.
    """
    valid = minute_close.notna() & (minute_suspend != 1)
    triggered = compute_first_threshold_cross_indicator(minute_close, valid, threshold)
    return triggered, valid


def aggregate_triggers_by_sector(
    stock_triggered: pd.DataFrame, stock_valid: pd.DataFrame, sector_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse per-STOCK leader triggers into per-SECTOR ones: sector S fires at bar t if
    ANY of its member stocks fired at t. Returns panels whose COLUMNS ARE SECTOR NAMES.

    This trades a sharper signal for far more of it, and is the right shape when the
    hypothesis is "this sector is moving" rather than "this particular bellwether leads
    that particular follower":

      - every member's events are pooled, so one hypothesis has thousands of observations
        instead of the few dozen a single stock accumulates in a short window;
      - the family shrinks from (N leaders x N followers) to (1 x N), which drops the
        multiple-testing bar substantially - with ~500 stocks in a sector that is 250k
        hypotheses versus 500.

    Deduped to at most one trigger per sector per day, the same way the per-stock
    detectors are: a sector where six stocks rip in the same session is one event, not
    six, or the pooled trigger count would just measure how broad the move was.

    A sector is `valid` at a bar when any member is - i.e. the sector is observable.

    IMPORTANT: pair this with `mask_self_triggers`. Without it, a stock that triggered its
    own sector is still measured as a follower of that trigger, which is the stock
    predicting itself (momentum), not spillover.
    """
    codes = [c for c in stock_triggered.columns if sector_map.get(c)]
    sectors = sorted({sector_map[c] for c in codes})
    dates = stock_triggered.index.normalize()

    triggered, valid = {}, {}
    for sector in sectors:
        members = [c for c in codes if sector_map[c] == sector]
        any_fired = stock_triggered[members].any(axis=1)
        cum = any_fired.groupby(dates).cumsum()
        triggered[sector] = any_fired & (cum == 1)
        valid[sector] = stock_valid[members].any(axis=1)

    return (pd.DataFrame(triggered, index=stock_triggered.index),
            pd.DataFrame(valid, index=stock_triggered.index))


def mask_self_triggers(
    follower_outcome: pd.DataFrame, follower_valid: pd.DataFrame, stock_triggered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark a follower invalid at any bar where IT was one of the stocks that triggered.

    Required with `aggregate_triggers_by_sector`: the sector-level leader includes the
    follower itself, so without this a stock that just surged would be counted as
    "following" the sector move it caused - measuring its own momentum, not spillover.

    This also drops bars where a DIFFERENT member triggered simultaneously, which loses a
    little data but errs on the conservative side.
    """
    fired = stock_triggered.reindex(columns=follower_valid.columns, fill_value=False).astype(bool)
    valid = follower_valid & ~fired
    return follower_outcome & valid, valid


def build_follower_frames(
    minute_close: pd.DataFrame, minute_suspend: pd.DataFrame, lag_bars: int,
    mode: str = "absolute", threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (follower_outcome, follower_valid) panels: did the price rise from
    bar t to bar t + lag_bars, without crossing into the next trading day? Indexed by
    the STARTING bar t, already forward-looking - see
    leadlag.factor.compute_pairwise_stats_same_row's docstring for why this must be
    paired with the "same_row" mining/validation functions, not the shifted ones.
    """
    valid_raw = minute_close.notna() & (minute_suspend != 1)
    return compute_forward_outcome(minute_close, valid_raw, lag_bars, mode=mode, threshold=threshold)


def build_follower_frames_overnight(
    minute_close: pd.DataFrame, minute_suspend: pd.DataFrame, exit_at: str = "next_open",
    mode: str = "absolute", threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (follower_outcome, follower_valid) panels for a T+1-executable hold: did
    the price rise from bar t to the NEXT TRADING DAY's exit bar? Also indexed by the
    starting bar t, so it pairs with the same "same_row" mining/validation functions as
    `build_follower_frames`.

    Use this whenever the backtest holds overnight - mining `build_follower_frames`'
    same-day outcome and then trading an overnight hold validates one hypothesis and
    trades another. See `intraday.data.compute_overnight_outcome`.
    """
    valid_raw = minute_close.notna() & (minute_suspend != 1)
    return compute_overnight_outcome(
        minute_close, valid_raw, exit_at=exit_at, mode=mode, threshold=threshold
    )
