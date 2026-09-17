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
from intraday.backtest import IntradayBacktestConfig, build_intraday_scores, simulate_intraday_portfolio
from intraday.event import build_leader_frames
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
    return p.parse_args()


def load_panels(args, pairs: pd.DataFrame, period: str):
    """Returns (leader_triggered, minute_close)."""
    if args.source == "synthetic":
        minute_close, minute_suspend, daily_close, _ = idd.make_synthetic_intraday_market()
        leader_triggered, _ = build_leader_frames(minute_close, minute_suspend, daily_close)
        return leader_triggered, minute_close
    if args.source == "csv":
        if not (args.close_csv and args.daily_close_csv):
            raise SystemExit("--close-csv and --daily-close-csv are required for --source csv")
        minute_close = pd.read_csv(args.close_csv, index_col=0, parse_dates=True).sort_index()
        daily_close = pd.read_csv(args.daily_close_csv, index_col=0, parse_dates=True).sort_index()
        minute_suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=minute_close.index, columns=minute_close.columns)
        leader_triggered, _ = build_leader_frames(minute_close, minute_suspend, daily_close)
        return leader_triggered, minute_close
    if args.source == "xtdata":
        stock_list = sorted(set(pairs["leader"]) | set(pairs["follower"]))
        minute_close, minute_suspend, daily_close = idd.fetch_intraday_panels_xtdata(
            stock_list, start_time=args.start, end_time=args.end, period=period, download=False,
        )
        leader_triggered, _ = build_leader_frames(minute_close, minute_suspend, daily_close)
        return leader_triggered, minute_close
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

    leader_triggered, minute_close = load_panels(args, pairs, meta.get("period", "5m"))

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
    )
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
