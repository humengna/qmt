"""Leader-side and follower-side signal construction for same-day intraday spillover.

All of the actual statistics (FDR-controlled mining, out-of-sample re-validation) are
reused directly from `leadlag.factor`'s same-row-aligned entry points
(`compute_pairwise_stats_same_row`, `validate_out_of_sample_same_row`) - this module
only builds the two boolean panels those functions need.
"""
from __future__ import annotations

import pandas as pd

from .data import broadcast_prev_close_to_bars, compute_first_touch_indicator, compute_forward_outcome


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
