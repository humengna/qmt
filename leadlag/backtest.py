"""Vectorized-signal, event-driven-fill backtest for a mined lead-lag pair table.

Timing convention (matches how the QMT strategy in strategy/leadlag_strategy.py
actually executes): a pair fires when its leader is "up" at day t's close; the
follower trade is sized and entered at day (t + lag)'s OPEN, then marked to market
at that day's close. Position sizing uses the previous day's ending equity, never
the current day's own close, so there is no look-ahead in how much is committed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    lag: int = 1
    top_k: int = 10
    holding_period: int = 1
    commission_bps: float = 3.0    # per side, in bps of trade value
    stamp_tax_bps: float = 5.0     # sell side only (A-share stamp duty)
    slippage_bps: float = 10.0     # per side, crude proxy for adverse selection on the open
    initial_capital: float = 1_000_000.0


def build_follower_scores(
    up: pd.DataFrame, pairs: pd.DataFrame, lag: int = 1, weight_col: str = "z"
) -> pd.DataFrame:
    """Turn a mined pair table into a T x follower daily trigger score.

    score.loc[day, follower] = sum of `weight_col` over every pair whose leader was
    "up" `lag` trading days before `day`, i.e. the score actionable on that day.
    """
    leaders = [c for c in pairs["leader"].unique() if c in up.columns]
    followers = sorted(pairs["follower"].unique())
    if not leaders or not followers:
        return pd.DataFrame(0.0, index=up.index, columns=followers)

    weight = (
        pairs.pivot_table(index="leader", columns="follower", values=weight_col, aggfunc="sum")
        .reindex(index=leaders, columns=followers)
        .fillna(0.0)
    )
    trigger = up[leaders].to_numpy(dtype=np.float64) @ weight.to_numpy(dtype=np.float64)
    score = pd.DataFrame(trigger, index=up.index, columns=followers).shift(lag)
    return score


def simulate_portfolio(
    score: pd.DataFrame, open_px: pd.DataFrame, close_px: pd.DataFrame, cfg: BacktestConfig
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate a top-K, equal-weight, fixed-holding-period long-only portfolio.

    Returns (equity curve indexed like `score`, trade log DataFrame).
    """
    dates = score.index
    one_side_cost = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0
    sell_cost = one_side_cost + cfg.stamp_tax_bps / 10_000.0

    cash = cfg.initial_capital
    prev_equity = cfg.initial_capital
    positions: dict[str, dict] = {}  # code -> {shares, entry_idx, entry_price}
    equity_curve = []
    trade_log = []

    for i, day in enumerate(dates):
        # 1) exit positions that have reached their holding period, at today's open
        to_close = [c for c, pos in positions.items() if i - pos["entry_idx"] >= cfg.holding_period]
        for code in to_close:
            pos = positions.pop(code)
            px = _lookup(open_px, day, code, default=pos["entry_price"])
            proceeds = pos["shares"] * px * (1 - sell_cost)
            cash += proceeds
            entry_cost = pos["shares"] * pos["entry_price"] * (1 + one_side_cost)
            trade_log.append({
                "day": day, "code": code, "side": "sell",
                "shares": pos["shares"], "price": px, "pnl": proceeds - entry_cost,
            })

        # 2) enter today's top-K positive-score followers not already held, sized off
        #    yesterday's ending equity (no look-ahead into today's own close)
        if day in score.index:
            today_scores = score.loc[day].dropna()
            today_scores = today_scores[today_scores > 0].sort_values(ascending=False)
        else:
            today_scores = pd.Series(dtype=float)

        free_slots = cfg.top_k - len(positions)
        if free_slots > 0 and not today_scores.empty:
            budget_per_name = prev_equity / cfg.top_k
            picked = 0
            for code in today_scores.index:
                if picked >= free_slots:
                    break
                if code in positions:
                    continue
                px = _lookup(open_px, day, code)
                if px is None or px <= 0:
                    continue
                shares = int(budget_per_name // (px * 100)) * 100  # round lot = 100 shares
                cost = shares * px * (1 + one_side_cost)
                if shares <= 0 or cost > cash:
                    continue
                cash -= cost
                positions[code] = {"shares": shares, "entry_idx": i, "entry_price": px}
                trade_log.append({
                    "day": day, "code": code, "side": "buy",
                    "shares": shares, "price": px, "pnl": 0.0,
                })
                picked += 1

        # 3) mark remaining positions to market at today's close
        equity = cash
        for code, pos in positions.items():
            px = _lookup(close_px, day, code, default=pos["entry_price"])
            equity += pos["shares"] * px
        equity_curve.append(equity)
        prev_equity = equity

    return pd.Series(equity_curve, index=dates, name="equity"), pd.DataFrame(trade_log)


def _lookup(panel: pd.DataFrame, day, code, default=None):
    if code not in panel.columns or day not in panel.index:
        return default
    val = panel.at[day, code]
    return val if pd.notna(val) and val > 0 else default
