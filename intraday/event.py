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
