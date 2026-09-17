"""Backtest a mined lead-lag pair table against a price panel.

Usage::

    python research/run_backtest.py --pairs research/output/pairs.csv --source synthetic
    python research/run_backtest.py --pairs research/output/pairs.csv --source xtdata --start 20180101
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
from leadlag.factor import compute_returns, compute_up_indicator
from leadlag.metrics import performance_summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--close-csv")
    p.add_argument("--open-csv")
    p.add_argument("--sectors", nargs="*", default=["沪深A股"])
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--holding-period", type=int, default=1)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--equity-csv", default="research/output/equity_curve.csv")
    p.add_argument("--trades-csv", default="research/output/trades.csv",
                    help="per-trade log; inspect this first when the equity curve is "
                         "negative despite the pairs having passed out-of-sample FDR.")
    p.add_argument("--eval-start", default="use-meta",
                    help="Only report performance from this date onward (YYYY-MM-DD). "
                         "Default 'use-meta' restricts to the mining run's held-out test "
                         "window (see pairs.meta.json's test_start) so the headline numbers "
                         "aren't inflated by the same days the pairs were mined on. "
                         "Pass 'none' to use the full price history instead.")
    return p.parse_args()


def load_panels(args):
    if args.source == "synthetic":
        open_px, close_px, _ = ld.make_synthetic_market()
        return open_px, close_px
    if args.source == "csv":
        if not args.close_csv:
            raise SystemExit("--close-csv is required for --source csv")
        return ld.load_price_panels_csv(args.close_csv, args.open_csv)
    if args.source == "xtdata":
        stock_list = ld.get_full_market_stock_list_xtdata(tuple(args.sectors))
        return ld.fetch_price_panels_xtdata(stock_list, start_time=args.start, end_time=args.end)
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    pairs = pd.read_csv(args.pairs)
    if pairs.empty:
        raise SystemExit(f"{args.pairs} has no pairs to trade")

    meta_path = Path(args.pairs).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "mode": "excess", "threshold": 0.0, "lag": 1,
    }

    open_px, close_px = load_panels(args)
    returns = compute_returns(close_px)
    up, _ = compute_up_indicator(returns, mode=meta["mode"], threshold=meta["threshold"])

    weight_col = "oos_z" if "oos_z" in pairs.columns else "z"
    score = build_follower_scores(up, pairs, lag=meta["lag"], weight_col=weight_col)

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
