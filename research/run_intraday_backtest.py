"""Backtest a mined same-day intraday-spillover pair table.

Usage::

    python research/run_intraday_backtest.py --pairs research/output/intraday_pairs.csv --source synthetic
    python research/run_intraday_backtest.py --pairs research/output/intraday_pairs.csv --source xtdata \\
        --start 20240101 --end 20240601
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from intraday import data as idd
from intraday.backtest import (
    IntradayBacktestConfig,
    build_intraday_scores,
    simulate_intraday_portfolio,
    simulate_overnight_portfolio,
)
from intraday.event import build_leader_frames, build_leader_frames_surge
from leadlag.metrics import performance_summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--close-csv")
    p.add_argument("--daily-close-csv")
    p.add_argument("--suspend-csv")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--equity-csv", default="research/output/intraday_equity_curve.csv")
    p.add_argument("--trades-csv", default="research/output/intraday_trades.csv",
                    help="per-trade log (bar, code, side, shares, price, pnl); the "
                         "first thing to inspect when the equity curve is negative "
                         "despite the pairs having passed out-of-sample FDR - it tells "
                         "you whether a small real edge is being eaten by costs, or "
                         "specific followers are just losing money outright.")
    p.add_argument("--eval-start", default="use-meta",
                    help="Only report performance from this bar timestamp onward. "
                         "Default 'use-meta' restricts to the mining run's held-out "
                         "test window. Pass 'none' for full history.")
    p.add_argument("--commission-bps", type=float, default=3.0,
                    help="per side, in bps of trade value. Set this to your broker's actual "
                         "rate - discount rates around 1bp are common.")
    p.add_argument("--slippage-bps", type=float, default=10.0,
                    help="per side. NOT a fee - this is the cost of crossing the spread, and "
                         "it stays real even with zero commission: fills here are modelled at "
                         "the bar's CLOSE, while a live market order pays the ask to buy and "
                         "the bid to sell. Running two cost points and solving for the "
                         "break-even cost beats arguing about what this number should be.")
    p.add_argument("--stamp-tax-bps", type=float, default=5.0,
                    help="sell side only. 5.0 is correct for STOCKS - unlike ETFs, which are "
                         "exempt from 印花税 (see research/run_etf_backtest.py).")
    p.add_argument("--download", action="store_true",
                    help="(xtdata only) fetch this period's history for the pair table's "
                         "symbols first. Needed for a FORWARD test whose --start is past the "
                         "mining window, since those bars were never downloaded.")
    return p.parse_args()


def _leader_triggered(meta: dict, minute_close, minute_suspend, daily_close):
    """Rebuild the SAME leader trigger the mining run used, per its meta.json."""
    if meta.get("trigger", "limitup") == "surge":
        triggered, _ = build_leader_frames_surge(
            minute_close, minute_suspend,
            threshold=meta.get("leader_threshold", 0.02), window_bars=meta.get("surge_window", 5),
        )
    else:
        triggered, _ = build_leader_frames(
            minute_close, minute_suspend, daily_close, tolerance=meta.get("tolerance", 0.003)
        )
    return triggered


def load_panels(args, pairs: pd.DataFrame, meta: dict):
    """Returns (leader_triggered, minute_close)."""
    period = meta.get("period", "5m")
    surge = meta.get("trigger", "limitup") == "surge"
    if args.source == "synthetic":
        if surge:
            minute_close, minute_suspend, _ = idd.make_synthetic_surge_market(
                threshold=meta.get("leader_threshold", 0.02), window_bars=meta.get("surge_window", 5),
            )
            daily_close = None
        else:
            minute_close, minute_suspend, daily_close, _ = idd.make_synthetic_intraday_market()
        return _leader_triggered(meta, minute_close, minute_suspend, daily_close), minute_close
    if args.source == "csv":
        if not args.close_csv:
            raise SystemExit("--close-csv is required for --source csv")
        if not surge and not args.daily_close_csv:
            raise SystemExit("--daily-close-csv is required for --source csv with a limitup-trigger pair table")
        minute_close = pd.read_csv(args.close_csv, index_col=0, parse_dates=True).sort_index()
        daily_close = pd.read_csv(args.daily_close_csv, index_col=0, parse_dates=True).sort_index() \
            if args.daily_close_csv else None
        minute_suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=minute_close.index, columns=minute_close.columns)
        return _leader_triggered(meta, minute_close, minute_suspend, daily_close), minute_close
    if args.source == "xtdata":
        stock_list = sorted(set(pairs["leader"]) | set(pairs["follower"]))
        if surge:
            minute_close, minute_suspend = idd.fetch_intraday_close_panels_xtdata(
                stock_list, start_time=args.start, end_time=args.end, period=period,
                download=args.download,
            )
            daily_close = None
        else:
            minute_close, minute_suspend, daily_close = idd.fetch_intraday_panels_xtdata(
                stock_list, start_time=args.start, end_time=args.end, period=period,
                download=args.download,
            )
        if minute_close.empty or minute_close.shape[1] == 0:
            raise SystemExit(
                f"no {period} bars for the pair table's {len(stock_list)} symbol(s) over "
                f"{args.start or '(open)'}..{args.end or '(open)'}. If this is a FORWARD test over "
                f"a range later than the mining window, that history was never downloaded - "
                f"re-run with --download."
            )
        return _leader_triggered(meta, minute_close, minute_suspend, daily_close), minute_close
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    pairs = pd.read_csv(args.pairs)
    if pairs.empty:
        raise SystemExit(f"{args.pairs} has no pairs to trade")

    meta_path = Path(args.pairs).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "lag_bars": 6, "period": "5m", "test_start": None,
    }

    leader_triggered, minute_close = load_panels(args, pairs, meta)

    weight_col = "oos_z" if "oos_z" in pairs.columns else "z"
    score = build_intraday_scores(leader_triggered, pairs, weight_col=weight_col)

    eval_start = args.eval_start
    if eval_start == "use-meta":
        eval_start = meta.get("test_start")
    if eval_start and eval_start.lower() != "none":
        cutoff = pd.Timestamp(eval_start)
        score, minute_close = (df[df.index >= cutoff] for df in (score, minute_close))
        print(f"evaluating from {cutoff} onward ({len(score)} bars) to avoid scoring "
              f"the same bars the pairs were mined on")

    cfg = IntradayBacktestConfig(
        lag_bars=meta["lag_bars"], top_k=args.top_k, initial_capital=args.initial_capital,
        commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
        stamp_tax_bps=args.stamp_tax_bps,
    )
    # the hold is whatever the mining VALIDATED - trading a different one would make the
    # backtest test a hypothesis the statistics never checked
    if meta.get("hold", "same-day") == "overnight":
        exit_at = meta.get("exit_at", "next_open")
        print(f"holding overnight, exiting at the next trading day's {exit_at} (T+1 executable)")
        equity, trades = simulate_overnight_portfolio(score, minute_close, cfg, exit_at=exit_at)
    else:
        print(f"holding {cfg.lag_bars} bars within the session - NOT executable on A-share "
              f"equities (T+1). Re-mine with --hold overnight for a tradeable variant.")
        equity, trades = simulate_intraday_portfolio(score, minute_close, cfg)

    bars_per_day = minute_close.groupby(minute_close.index.normalize()).size().median()
    summary = performance_summary(equity, periods_per_year=int(252 * bars_per_day))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_path = Path(args.equity_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_path)

    trades_path = Path(args.trades_csv)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_path, index=False)
    print(f"wrote equity curve to {out_path} and trade log to {trades_path} ({len(trades)} trades)")

    sells = trades[trades["side"] == "sell"]
    if not sells.empty:
        by_code = sells.groupby("code")["pnl"].agg(["sum", "count"]).sort_values("sum")
        print("\nP&L by follower (worst to best - inspect the biggest losers first):")
        print(by_code.to_string())


if __name__ == "__main__":
    main()
