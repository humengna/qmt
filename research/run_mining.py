"""End-to-end lead-lag pair mining pipeline.

Usage examples
--------------
Smoke test on fabricated data (no QMT / network needed)::

    python research/run_mining.py --source synthetic --output research/output/pairs.csv

Mine on real data pulled from a running QMT/MiniQMT terminal via xtquant.xtdata::

    python research/run_mining.py --source xtdata --start 20180101 \\
        --output research/output/pairs.csv

Mine on your own wide-format CSV price panels (date index, one column per stock)::

    python research/run_mining.py --source csv --close-csv close.csv --open-csv open.csv \\
        --output research/output/pairs.csv

The pipeline: split history into a train/test window -> mine candidate pairs on the
train window with FDR control -> re-validate every candidate on the untouched test
window -> keep only pairs whose lift survives out-of-sample -> trim to a symbol
budget the live strategy can actually subscribe to. See leadlag/factor.py and
README.md for why every one of these steps exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leadlag import data as ld
from leadlag.factor import (
    MiningConfig,
    compute_returns,
    compute_up_indicator,
    mine_lead_lag_pairs,
    select_for_deployment,
    validate_out_of_sample,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--sectors", nargs="*", default=["沪深A股"])
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="")
    p.add_argument("--close-csv")
    p.add_argument("--open-csv")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess",
                    help="'excess' nets out each day's cross-sectional median return "
                         "before deciding 'up', to avoid mistaking common market moves "
                         "for a leader-follower relationship (recommended).")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--min-obs", type=int, default=60)
    p.add_argument("--alpha", type=float, default=0.01, help="BH-FDR level for in-sample mining")
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-symbols", type=int, default=500,
                    help="live-subscribe budget; see docs/QMT_API_NOTES.md")
    p.add_argument("--min-oos-n", type=int, default=20)
    p.add_argument("--output", default="research/output/pairs.csv")
    return p.parse_args()


def load_panels(args):
    if args.source == "synthetic":
        open_px, close_px, injected = ld.make_synthetic_market()
        print(f"[synthetic] injected {len(injected)} true lead-lag pairs: {injected}")
        return open_px, close_px
    if args.source == "csv":
        if not args.close_csv:
            raise SystemExit("--close-csv is required for --source csv")
        return ld.load_price_panels_csv(args.close_csv, args.open_csv)
    if args.source == "xtdata":
        stock_list = ld.get_full_market_stock_list_xtdata(tuple(args.sectors))
        print(f"universe size: {len(stock_list)}")
        return ld.fetch_price_panels_xtdata(stock_list, start_time=args.start, end_time=args.end)
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    open_px, close_px = load_panels(args)

    returns = compute_returns(close_px)
    split = int(len(returns) * args.train_frac)
    train_returns, test_returns = returns.iloc[:split], returns.iloc[split:]
    print(f"train window: {train_returns.index.min()} .. {train_returns.index.max()} "
          f"({len(train_returns)} days)")
    print(f"test window:  {test_returns.index.min()} .. {test_returns.index.max()} "
          f"({len(test_returns)} days)")

    cfg = MiningConfig(lag=args.lag, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    up_train, valid_train = compute_up_indicator(train_returns, mode=args.mode, threshold=args.threshold)
    print(f"mining {up_train.shape[1]} symbols x {up_train.shape[0]} train days ...")
    candidates = mine_lead_lag_pairs(up_train, valid_train, cfg)
    print(f"{len(candidates)} candidate pairs survive FDR-controlled in-sample mining (alpha={cfg.alpha})")

    up_test, valid_test = compute_up_indicator(test_returns, mode=args.mode, threshold=args.threshold)
    validated = validate_out_of_sample(up_test, valid_test, candidates, cfg)
    n_survive = ((validated["oos_lift"] > 0) & (validated["oos_n"] >= args.min_oos_n)).sum()
    print(f"{n_survive} pairs keep a positive lift out-of-sample")

    deployable = select_for_deployment(
        validated, max_unique_symbols=args.max_symbols, min_oos_n=args.min_oos_n
    )
    n_symbols = len(set(deployable["leader"]) | set(deployable["follower"])) if len(deployable) else 0
    print(f"{len(deployable)} pairs kept for deployment ({n_symbols} unique symbols, "
          f"budget={args.max_symbols})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deployable.to_csv(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "mode": args.mode,
        "threshold": args.threshold,
        "lag": args.lag,
        "test_start": str(test_returns.index.min()) if len(test_returns) else None,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
