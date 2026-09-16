"""Backtest a mined limit-up-trigger pair table against a price panel.

Usage::

    python research/run_limitup_backtest.py --pairs research/output/limitup_pairs.csv --source synthetic
    python research/run_limitup_backtest.py --pairs research/output/limitup_pairs.csv --source xtdata --start 20180101
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from leadlag import data as ld
from leadlag.backtest import BacktestConfig, build_follower_scores, simulate_portfolio
from leadlag.metrics import performance_summary
from limitup import data as lud
from limitup.event import build_leader_frames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--close-csv")
    p.add_argument("--preclose-csv")
    p.add_argument("--suspend-csv")
    p.add_argument("--open-csv")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--holding-period", type=int, default=1)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--equity-csv", default="research/output/limitup_equity_curve.csv")
    p.add_argument("--eval-start", default="use-meta",
                    help="Only report performance from this date onward. Default "
                         "'use-meta' restricts to the mining run's held-out test window "
                         "(pairs.meta.json's test_start). Pass 'none' for full history.")
    return p.parse_args()


def load_panels(args, pairs: pd.DataFrame):
    """Returns (leader_up, leader_valid, open_px, close_px)."""
    if args.source == "synthetic":
        raw_close, raw_preclose, suspend, adj_close, _ = lud.make_synthetic_limitup_market()
        leader_up, leader_valid = build_leader_frames(raw_close, raw_preclose, suspend)
        open_px = adj_close.shift(1).bfill()  # synthetic market has no separate open series
        return leader_up, leader_valid, open_px, adj_close
    if args.source == "csv":
        if not (args.close_csv and args.preclose_csv):
            raise SystemExit("--close-csv and --preclose-csv are required for --source csv")
        raw_close, raw_preclose = ld.load_price_panels_csv(args.close_csv, args.preclose_csv)
        suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=raw_close.index, columns=raw_close.columns)
        leader_up, leader_valid = build_leader_frames(raw_close, raw_preclose, suspend)
        open_px, close_px = ld.load_price_panels_csv(args.close_csv, args.open_csv)
        return leader_up, leader_valid, open_px, close_px
    if args.source == "xtdata":
        # Only fetch what this pair table actually needs, not the whole market.
        stock_list = sorted(set(pairs["leader"]) | set(pairs["follower"]))
        raw_close, raw_preclose, suspend, _ = lud.fetch_limitup_panels_xtdata(
            stock_list, start_time=args.start, end_time=args.end, download=False,
        )
        leader_up, leader_valid = build_leader_frames(raw_close, raw_preclose, suspend)
        open_px, close_px = ld.fetch_price_panels_xtdata(
            stock_list, start_time=args.start, end_time=args.end, download=False,
        )
        return leader_up, leader_valid, open_px, close_px
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    pairs = pd.read_csv(args.pairs)
    if pairs.empty:
        raise SystemExit(f"{args.pairs} has no pairs to trade")

    meta_path = Path(args.pairs).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"lag": 1, "test_start": None}

    leader_up, leader_valid, open_px, close_px = load_panels(args, pairs)

    weight_col = "oos_z" if "oos_z" in pairs.columns else "z"
    score = build_follower_scores(leader_up, pairs, lag=meta["lag"], weight_col=weight_col)

    eval_start = args.eval_start
    if eval_start == "use-meta":
        eval_start = meta.get("test_start")
    if eval_start and eval_start.lower() != "none":
        cutoff = pd.Timestamp(eval_start)
        score, open_px, close_px = (df[df.index >= cutoff] for df in (score, open_px, close_px))
        print(f"evaluating from {cutoff.date()} onward ({len(score)} days) to avoid scoring "
              f"the same days the pairs were mined on")

    cfg = BacktestConfig(
        lag=meta["lag"], top_k=args.top_k, holding_period=args.holding_period,
        initial_capital=args.initial_capital,
    )
    equity, trades = simulate_portfolio(score, open_px, close_px, cfg)

    summary = performance_summary(equity)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_path = Path(args.equity_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_path)
    print(f"wrote equity curve to {out_path} ({len(trades)} trades)")


if __name__ == "__main__":
    main()
