"""Leader-side signal construction for the limit-up-trigger hypothesis.

Everything else (FDR-controlled mining, out-of-sample validation, the backtest
engine) is reused directly from `leadlag.factor` / `leadlag.backtest` via their
asymmetric entry points (`compute_pairwise_stats_asymmetric`,
`validate_out_of_sample_asymmetric`) - the leader's trigger (closed at limit-up) and
the follower's outcome (regular up) are different signals here, unlike leadlag's own
symmetric use of "up" on both sides. See research/run_limitup_mining.py for how the
pieces fit together end to end.
"""
from __future__ import annotations

import pandas as pd

from .data import compute_limit_up_indicator


def build_leader_frames(
    raw_close: pd.DataFrame, raw_preclose: pd.DataFrame, suspend: pd.DataFrame, tolerance: float = 0.003,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (leader_triggered, leader_valid) panels: did the stock close at its
    daily price-limit that day? Requires UNADJUSTED close/preclose (dividend_type='none'
    when fetching - see `limitup.data.fetch_limitup_panels_xtdata`).
    """
    valid = raw_close.notna() & raw_preclose.notna() & (suspend != 1)
    triggered = compute_limit_up_indicator(raw_close, raw_preclose, valid, tolerance=tolerance)
    return triggered, valid
