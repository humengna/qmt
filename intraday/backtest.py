"""Intraday backtest engine for same-day spillover signals.

Unlike leadlag's daily backtest (a signal at today's close is acted on at TOMORROW's
open), a signal here fires and is acted on WITHIN THE SAME TRADING DAY: a leader's
first-touch at bar t is followed by an entry at (approximately) that same bar, held
for exactly `lag_bars` bars - the same window `intraday.event.build_follower_frames`
validated - and never past that trading day's last bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IntradayBacktestConfig:
    lag_bars: int = 6
    top_k: int = 5
    commission_bps: float = 3.0     # per side, in bps of trade value
    stamp_tax_bps: float = 5.0      # sell side only (A-share stamp duty)
    slippage_bps: float = 10.0      # per side; intraday fills are more adverse-selected
    initial_capital: float = 1_000_000.0


def build_intraday_scores(
    leader_triggered: pd.DataFrame, pairs: pd.DataFrame, weight_col: str = "z",
) -> pd.DataFrame:
    """score[t, follower] = sum of `weight_col` over pairs whose leader triggered
    EXACTLY at bar t. No shift, unlike `leadlag.backtest.build_follower_scores`: the
    entry happens at (approximately) the bar the trigger fires, not `lag` bars later -
    the "later" part is the HOLDING period, applied in `simulate_intraday_portfolio`.
    """
    leaders = [c for c in pairs["leader"].unique() if c in leader_triggered.columns]
    followers = sorted(pairs["follower"].unique())
    if not leaders or not followers:
        return pd.DataFrame(0.0, index=leader_triggered.index, columns=followers)

    weight = (
        pairs.pivot_table(index="leader", columns="follower", values=weight_col, aggfunc="sum")
        .reindex(index=leaders, columns=followers)
        .fillna(0.0)
    )
    score = leader_triggered[leaders].to_numpy(dtype=np.float64) @ weight.to_numpy(dtype=np.float64)
    return pd.DataFrame(score, index=leader_triggered.index, columns=followers)


def simulate_intraday_portfolio(
    score: pd.DataFrame, close_px: pd.DataFrame, cfg: IntradayBacktestConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Enter a follower at the bar its score turns positive, exit exactly
    `cfg.lag_bars` bars later. A trigger bar without `cfg.lag_bars` of room left in
    its trading day is skipped entirely (not truncated to a shorter hold) - this keeps
    every realized trade matching the exact window `filter_oos_significant` actually
    validated, rather than quietly trading an un-validated shorter variant near the
    close. Position sizing uses equity as of the LAST bar marked to market, never the
    entry bar's own close (no look-ahead in sizing).
    """
    bar_times = score.index
    dates = bar_times.normalize()
    one_side_cost = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0
    sell_cost = one_side_cost + cfg.stamp_tax_bps / 10_000.0

    last_bar_idx_of_day = pd.Series(np.arange(len(bar_times)), index=bar_times).groupby(dates).transform("max")

    cash = cfg.initial_capital
    prev_equity = cfg.initial_capital
    positions: dict[str, dict] = {}  # code -> {shares, exit_idx, entry_price}
    equity_curve = []
    trade_log = []

    for i, bar_time in enumerate(bar_times):
        # 1) exit positions scheduled to close at or before this bar
        to_close = [c for c, pos in positions.items() if pos["exit_idx"] <= i]
        for code in to_close:
            pos = positions.pop(code)
            px = _lookup(close_px, bar_time, code, default=pos["entry_price"])
            proceeds = pos["shares"] * px * (1 - sell_cost)
            cash += proceeds
            entry_cost = pos["shares"] * pos["entry_price"] * (1 + one_side_cost)
            trade_log.append({
                "bar": bar_time, "code": code, "side": "sell",
                "shares": pos["shares"], "price": px, "pnl": proceeds - entry_cost,
            })

        # 2) enter today's top-K positive-score followers, only if the full
        #    lag_bars holding window still fits in today's session
        exit_idx = i + cfg.lag_bars
        has_room = exit_idx <= last_bar_idx_of_day.iloc[i]
        free_slots = cfg.top_k - len(positions)
        if free_slots > 0 and has_room and bar_time in score.index:
            today_scores = score.loc[bar_time].dropna()
            today_scores = today_scores[today_scores > 0].sort_values(ascending=False)
            if not today_scores.empty:
                budget_per_name = prev_equity / cfg.top_k
                picked = 0
                for code in today_scores.index:
                    if picked >= free_slots:
                        break
                    if code in positions:
                        continue
                    px = _lookup(close_px, bar_time, code)
                    if px is None or px <= 0:
                        continue
                    shares = int(budget_per_name // (px * 100)) * 100  # round lot = 100 shares
                    cost = shares * px * (1 + one_side_cost)
                    if shares <= 0 or cost > cash:
                        continue
                    cash -= cost
                    positions[code] = {"shares": shares, "exit_idx": exit_idx, "entry_price": px}
                    trade_log.append({
                        "bar": bar_time, "code": code, "side": "buy",
                        "shares": shares, "price": px, "pnl": 0.0,
                    })
                    picked += 1

        # 3) mark remaining positions to market at this bar's close
        equity = cash
        for code, pos in positions.items():
            px = _lookup(close_px, bar_time, code, default=pos["entry_price"])
            equity += pos["shares"] * px
        equity_curve.append(equity)
        prev_equity = equity

    return pd.Series(equity_curve, index=bar_times, name="equity"), pd.DataFrame(trade_log)


def _lookup(panel: pd.DataFrame, bar_time, code, default=None):
    if code not in panel.columns or bar_time not in panel.index:
        return default
    val = panel.at[bar_time, code]
    return val if pd.notna(val) and val > 0 else default
