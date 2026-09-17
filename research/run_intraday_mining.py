"""End-to-end mining pipeline for the SAME-DAY intraday spillover hypothesis.

Hypothesis under test: when a stock FIRST touches its daily price-limit (涨停) at
some specific intraday bar, other stocks in the SAME sector are more likely to rise
over the following `--lag-bars` bars, WITHIN THE SAME TRADING DAY, than their own
baseline propensity (same-day thematic spillover after a board is sealed). This
reuses leadlag's statistical machinery via the "same-row-aligned" entry points in
leadlag.factor (compute_pairwise_stats_same_row, validate_out_of_sample_same_row):
the follower's outcome here is already a forward-looking quantity computed AT the
trigger bar (see intraday/event.py), unlike leadlag/limitup's daily "check `lag` days
later" comparison.

DATA VOLUME WARNING: minute bars are far larger than daily bars (a trading day is
~240 one-minute bars / 48 five-minute bars). Scope `--sectors`/`--top-liquid` and the
date range down (a few hundred names, months rather than years) before pointing this
at the whole market for years of history - see intraday/README.md.

Usage examples
--------------
Smoke test on fabricated data (no QMT / network needed)::

    python research/run_intraday_mining.py --source synthetic --min-obs 15 \\
        --candidate-mode top-n --top-n 40 --output research/output/intraday_pairs.csv

Mine on real 5-minute bars from a running QMT/MiniQMT terminal, same-sector only
(the default) with ST names excluded (the default), over a SHORT date range::

    python research/run_intraday_mining.py --source xtdata --top-liquid 300 \\
        --start 20240101 --end 20240601 --output research/output/intraday_pairs.csv

As always, watch the "FDR-significant OUT-OF-SAMPLE" line, not the candidate count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from intraday import data as idd
from intraday.event import build_follower_frames, build_leader_frames
from leadlag import data as ld
from leadlag.factor import (
    MiningConfig,
    compute_pairwise_stats_same_row,
    filter_oos_significant,
    filter_significant_pairs,
    select_for_deployment,
    select_top_n_candidates,
    validate_out_of_sample_same_row,
)
from limitup import data as lud


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--sectors", nargs="*", default=["沪深A股"], help="universe sector(s), xtdata only")
    p.add_argument("--start", default="", help="leave short - see the data-volume warning above")
    p.add_argument("--end", default="")
    p.add_argument("--period", default="5m", help="bar granularity for get_market_data_ex/download_history_data")
    p.add_argument("--close-csv", help="wide CSV of minute close prices (xtdata alternative)")
    p.add_argument("--daily-close-csv", help="wide CSV of daily close prices (required with --close-csv)")
    p.add_argument("--suspend-csv")
    p.add_argument("--top-liquid", type=int, default=None,
                    help="(xtdata only) shrink the universe to the N stocks with the "
                         "highest median daily traded value before mining (ranked off "
                         "daily data - cheap even before any minute-bar download).")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.003)
    p.add_argument("--include-st", action="store_true")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--lag-bars", type=int, default=6,
                    help="how many bars ahead the follower's outcome is measured "
                         "(6 bars x 5m = 30 minutes); must stay within the same trading day")
    p.add_argument("--min-obs", type=int, default=15,
                    help="intraday first-touch events are rarer still than daily "
                         "limit-up days, so this defaults even lower than limitup's 30")
    p.add_argument("--candidate-mode", choices=["fdr", "top-n"], default="fdr")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--oos-alpha", type=float, default=0.05)
    p.add_argument("--min-oos-lift", type=float, default=0.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-symbols", type=int, default=500)
    p.add_argument("--min-oos-n", type=int, default=10)
    p.add_argument("--output", default="research/output/intraday_pairs.csv")
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--all-sectors", action="store_true")
    p.add_argument("--sector-names", nargs="*", default=None)
    return p.parse_args()


def print_diagnostics(stats: pd.DataFrame, cfg: MiningConfig) -> None:
    n_pairs = len(stats)
    print(f"\n--- diagnostics: {n_pairs} pairs cleared --min-obs={cfg.min_obs} in each branch ---")
    if n_pairs == 0:
        print("no pair has enough first-touch bars in both branches. Intraday triggers "
              "are rare - lower --min-obs, widen the date range, or check the universe "
              "actually has active/limit-up-prone names (a --top-liquid universe skews "
              "toward large caps that rarely hit limit-up at all).")
        return
    lift = stats["lift"].to_numpy()
    qs = [50, 90, 99, 99.9, 100]
    print("lift percentiles (P(follower up | leader triggered) - P(follower up | leader not)):")
    for q, v in zip(qs, np.percentile(lift, qs)):
        print(f"  p{q:>5}: {v:+.4f}")
    print(f"pairs with lift >= --min-lift={cfg.min_lift}: {(lift >= cfg.min_lift).sum()} (before FDR)")
    top = stats.sort_values("z", ascending=False).head(10)
    with pd.option_context("display.width", 120):
        print("top 10 pairs by z-score (NOT FDR-controlled - for inspection only):")
        print(top[["leader", "follower", "n_leader_up", "p_cond", "p_base", "lift", "z"]]
              .to_string(index=False))
    print("--- end diagnostics ---\n")


def load_panels(args):
    """Returns (minute_close, minute_suspend, daily_close, stock_list)."""
    if args.source == "synthetic":
        minute_close, minute_suspend, daily_close, injected = idd.make_synthetic_intraday_market(
            lag_bars=args.lag_bars
        )
        print(f"[synthetic] injected {len(injected)} true same-day trigger pairs: {injected}")
        return minute_close, minute_suspend, daily_close, list(minute_close.columns)
    if args.source == "csv":
        if not (args.close_csv and args.daily_close_csv):
            raise SystemExit("--close-csv and --daily-close-csv are required for --source csv")
        minute_close = pd.read_csv(args.close_csv, index_col=0, parse_dates=True).sort_index()
        daily_close = pd.read_csv(args.daily_close_csv, index_col=0, parse_dates=True).sort_index()
        minute_suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=minute_close.index, columns=minute_close.columns)
        return minute_close, minute_suspend, daily_close, list(minute_close.columns)
    if args.source == "xtdata":
        stock_list = ld.get_full_market_stock_list_xtdata(tuple(args.sectors))
        print(f"universe size: {len(stock_list)}")
        if not args.include_st:
            st_codes = lud.build_st_exclusion_set_xtdata(stock_list)
            stock_list = [c for c in stock_list if c not in st_codes]
            print(f"excluded {len(st_codes)} ST/*ST names, {len(stock_list)} remain")
        if args.top_liquid:
            stock_list = ld.rank_stocks_by_liquidity_xtdata(
                stock_list, start_time=args.start, end_time=args.end, top_n=args.top_liquid
            )
            print(f"kept top {len(stock_list)} by median daily traded value")
        minute_close, minute_suspend, daily_close = idd.fetch_intraday_panels_xtdata(
            stock_list, start_time=args.start, end_time=args.end, period=args.period,
            download=not args.no_download,
        )
        return minute_close, minute_suspend, daily_close, stock_list
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    minute_close, minute_suspend, daily_close, stock_list = load_panels(args)

    sector_map = None
    if not args.all_sectors:
        if args.source == "xtdata":
            sector_names = args.sector_names or ld.DEFAULT_SW_L1_SECTORS
            sector_map = ld.build_sector_map_xtdata(sector_names)
            mapped = sum(1 for c in stock_list if c in sector_map)
            print(f"sector map covers {mapped}/{len(stock_list)} stocks "
                  f"({len(set(sector_map.values()))} sectors used)")
        else:
            print("[intraday] --source is not xtdata: cannot build a real sector map, "
                  "--all-sectors behavior is used regardless.")

    split = int(len(minute_close) * args.train_frac)
    train_sl, test_sl = slice(0, split), slice(split, None)
    print(f"train window: {minute_close.index[0]} .. {minute_close.index[split - 1]} ({split} bars)")
    print(f"test window:  {minute_close.index[split]} .. {minute_close.index[-1]} "
          f"({len(minute_close) - split} bars)")

    cfg = MiningConfig(lag=args.lag_bars, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    leader_triggered_full, leader_valid_full = build_leader_frames(
        minute_close, minute_suspend, daily_close, tolerance=args.tolerance
    )
    follower_outcome_full, follower_valid_full = build_follower_frames(
        minute_close, minute_suspend, lag_bars=args.lag_bars, mode=args.mode, threshold=args.threshold
    )

    total_triggers = int(leader_triggered_full.iloc[train_sl].to_numpy().sum())
    print(f"mining {leader_triggered_full.shape[1]} symbols x {split} train bars "
          f"({total_triggers} total first-touch events across the universe) "
          f"{'(same-sector pairs only)' if sector_map else ''}...")

    stats = compute_pairwise_stats_same_row(
        leader_triggered_full.iloc[train_sl], leader_valid_full.iloc[train_sl],
        follower_outcome_full.iloc[train_sl], follower_valid_full.iloc[train_sl], cfg, sector_map,
    )
    if not args.no_diagnostics:
        print_diagnostics(stats, cfg)

    if args.candidate_mode == "fdr":
        candidates = filter_significant_pairs(stats, cfg)
        print(f"{len(candidates)} candidate pairs survive FDR-controlled in-sample mining (alpha={cfg.alpha})")
    else:
        candidates = select_top_n_candidates(stats, cfg, args.top_n)
        print(f"{len(candidates)} candidate pairs taken by rank (top-n={args.top_n}, no FDR)")

    validated = validate_out_of_sample_same_row(
        leader_triggered_full.iloc[test_sl], leader_valid_full.iloc[test_sl],
        follower_outcome_full.iloc[test_sl], follower_valid_full.iloc[test_sl], candidates, cfg,
    )
    n_positive_sign = ((validated["oos_lift"] > 0) & (validated["oos_n"] >= args.min_oos_n)).sum()
    print(f"{n_positive_sign} pairs merely keep a positive oos_lift sign (NOT a real test)")

    oos_significant = filter_oos_significant(
        validated, alpha=args.oos_alpha, min_oos_lift=args.min_oos_lift, min_oos_n=args.min_oos_n
    )
    print(f"{len(oos_significant)} pairs are FDR-significant OUT-OF-SAMPLE "
          f"(alpha={args.oos_alpha}) -- this is the real gate")

    deployable = select_for_deployment(
        oos_significant, max_unique_symbols=args.max_symbols, min_oos_n=args.min_oos_n
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
        "lag_bars": args.lag_bars,
        "tolerance": args.tolerance,
        "period": args.period,
        "test_start": str(minute_close.index[split]) if split < len(minute_close) else None,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
