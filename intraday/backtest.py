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


def _same_day_exit_rows(bar_times: pd.DatetimeIndex, lag_bars: int) -> np.ndarray:
    """exit_rows[i] = i + lag_bars, or -1 when that would spill past bar i's own trading
    day. A trigger bar without enough room left is skipped entirely rather than truncated
    to a shorter hold, so every realized trade matches the exact window
    `filter_oos_significant` validated instead of a quietly un-validated shorter variant.
    """
    n = len(bar_times)
    last_of_day = pd.Series(np.arange(n), index=bar_times).groupby(bar_times.normalize()).transform("max")
    exit_rows = np.arange(n) + lag_bars
    exit_rows[exit_rows >= n] = -1
    spills = (exit_rows > last_of_day.to_numpy()) & (exit_rows >= 0)
    exit_rows[spills] = -1
    return exit_rows


def _overnight_exit_rows(bar_times: pd.DatetimeIndex, exit_at: str = "next_open") -> np.ndarray:
    """exit_rows[i] = the NEXT trading day's first (or last) bar row, or -1 for bars on
    the panel's final day, which have no next session to exit into.

    Every position therefore spans a day boundary, which is what makes this schedule
    executable on A-share equities at all (T+1: a stock bought today cannot be sold
    today). Pair it with `intraday.event.build_follower_frames_overnight` so the mining
    validated the same hold the backtest trades.
    """
    if exit_at not in ("next_open", "next_close"):
        raise ValueError(f"unknown exit_at: {exit_at!r}, expected 'next_open' or 'next_close'")

    n = len(bar_times)
    dates = bar_times.normalize()
    rows = pd.Series(np.arange(n), index=bar_times)
    per_day = rows.groupby(dates).min() if exit_at == "next_open" else rows.groupby(dates).max()
    next_day_row = per_day.shift(-1)
    return next_day_row.reindex(dates).fillna(-1).to_numpy(dtype=np.int64)


def simulate_intraday_portfolio(
    score: pd.DataFrame, close_px: pd.DataFrame, cfg: IntradayBacktestConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Enter a follower at the bar its score turns positive, exit exactly `cfg.lag_bars`
    bars later, never holding overnight.

    NOT executable on A-share equities (T+1 forbids the same-day round trip) - use
    `simulate_overnight_portfolio` for those. This stays for T0-eligible instruments and
    for measuring the same-day effect as a research question.
    """
    return _simulate(score, close_px, cfg, _same_day_exit_rows(score.index, cfg.lag_bars))


def simulate_overnight_portfolio(
    score: pd.DataFrame, close_px: pd.DataFrame, cfg: IntradayBacktestConfig,
    exit_at: str = "next_open",
) -> tuple[pd.Series, pd.DataFrame]:
    """Enter a follower at the bar its score turns positive, exit on the NEXT trading
    day - the T+1-executable version of `simulate_intraday_portfolio`.

    `cfg.lag_bars` is unused here: the hold is defined by the calendar, not a bar count.
    """
    return _simulate(score, close_px, cfg, _overnight_exit_rows(score.index, exit_at))


def _simulate(
    score: pd.DataFrame, close_px: pd.DataFrame, cfg: IntradayBacktestConfig,
    exit_rows: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    """Shared engine: `exit_rows[i]` is the bar row a position entered at bar i must be
    closed on, or -1 if bar i cannot be entered at all. Position sizing uses equity as of
    the LAST bar marked to market, never the entry bar's own close (no look-ahead).
    """
    bar_times = score.index
    one_side_cost = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0
    sell_cost = one_side_cost + cfg.stamp_tax_bps / 10_000.0

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

        # 2) enter this bar's top-K positive-score followers, if it has a usable exit
        exit_idx = int(exit_rows[i])
        free_slots = cfg.top_k - len(positions)
        if free_slots > 0 and exit_idx > i and bar_time in score.index:
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
