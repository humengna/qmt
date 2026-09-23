"""End-to-end mining pipeline for the limit-up-trigger hypothesis.

Hypothesis under test: when a stock closes AT its daily price-limit (涨停), other
stocks in the SAME sector are more likely to be up `--lag` trading days later, beyond
their own baseline propensity to rise (thematic/sentiment spillover - "龙头带动板块
补涨"). This reuses leadlag's statistical machinery (matrix-multiplied two-proportion
z-test, BH-FDR, out-of-sample re-validation) via the asymmetric entry points in
leadlag.factor; the only new piece is `limitup.event.build_leader_frames`, which
turns raw close/preclose/suspend panels into a "closed at limit-up" boolean panel
using each stock's board-based price-limit percentage (see limitup/data.py) - no
minute/tick data required.

Usage examples
--------------
Smoke test on fabricated data (no QMT / network needed)::

    python research/run_limitup_mining.py --source synthetic --output research/output/limitup_pairs.csv

Mine on real data pulled from a running QMT/MiniQMT terminal via xtquant.xtdata,
restricted to same-sector pairs (the default) with ST names excluded (the default)::

    python research/run_limitup_mining.py --source xtdata --start 20180101 \\
        --output research/output/limitup_pairs.csv

Same as leadlag's run_mining.py, "candidates surviving in-sample" is NOT the answer -
watch the "FDR-significant OUT-OF-SAMPLE" line. See README.md's statistical-pitfalls
section; it applies here just as much as to the plain lead-lag factor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from leadlag import data as ld
from leadlag.factor import (
    MiningConfig,
    compute_pairwise_stats_asymmetric,
    compute_up_indicator,
    filter_oos_significant,
    filter_significant_pairs,
    select_for_deployment,
    select_top_n_candidates,
    validate_out_of_sample_asymmetric,
)
from limitup import data as lud
from limitup.event import build_leader_frames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--sectors", nargs="*", default=["沪深A股"], help="universe sector(s), xtdata only")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="")
    p.add_argument("--close-csv")
    p.add_argument("--preclose-csv")
    p.add_argument("--suspend-csv")
    p.add_argument("--adjclose-csv")
    p.add_argument("--top-liquid", type=int, default=None,
                    help="(xtdata only) shrink the universe to the N stocks with the "
                         "highest median daily traded value before mining.")
    p.add_argument("--no-download", action="store_true",
                    help="(xtdata only) skip download_history_data, read only cached data.")
    p.add_argument("--tolerance", type=float, default=0.003,
                    help="price tolerance for the close-vs-computed-limit-price check")
    p.add_argument("--include-st", action="store_true",
                    help="don't exclude currently ST/*ST-flagged names (excluded by default; "
                         "see limitup.data.build_st_exclusion_set_xtdata for the caveats)")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess",
                    help="follower 'up' definition; 'excess' nets out the day's cross-"
                         "sectional median return first (recommended, see leadlag).")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--min-obs", type=int, default=30,
                    help="limit-up days are much rarer than plain 'up' days, so this "
                         "defaults lower than leadlag's 60 - raise it if a run reports "
                         "few pairs clearing --min-obs even before any FDR filtering.")
    p.add_argument("--candidate-mode", choices=["fdr", "top-n"], default="fdr")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.01, help="BH-FDR level for in-sample mining")
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--oos-alpha", type=float, default=0.05,
                    help="BH-FDR level applied to the OUT-OF-SAMPLE p-values - the real "
                         "gate. See leadlag.factor.filter_oos_significant.")
    p.add_argument("--min-oos-lift", type=float, default=0.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-symbols", type=int, default=500,
                    help="live-subscribe budget; see docs/QMT_API_NOTES.md")
    p.add_argument("--min-oos-n", type=int, default=10)
    p.add_argument("--output", default="research/output/limitup_pairs.csv")
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--all-sectors", action="store_true",
                    help="test every cross-sector pair too, instead of the default "
                         "same-sector-only restriction (see leadlag's README for why "
                         "same-sector is both more plausible and statistically easier "
                         "to clear FDR/out-of-sample control).")
    p.add_argument("--sector-names", nargs="*", default=None,
                    help="override leadlag.data.DEFAULT_SW_L1_SECTORS; verify with "
                         "'python research/run_mining.py --source xtdata --list-sectors' first.")
    return p.parse_args()


def print_diagnostics(stats: pd.DataFrame, cfg: MiningConfig) -> None:
    n_pairs = len(stats)
    print(f"\n--- diagnostics: {n_pairs} pairs cleared --min-obs={cfg.min_obs} in each branch ---")
    if n_pairs == 0:
        print("no pair has >= min_obs (leader limit-up) days AND >= min_obs (leader "
              "valid-not-triggered) days. Limit-ups are rare - lower --min-obs, use a "
              "longer history, or check the universe actually has limit-up days at all "
              "(a --top-liquid universe skews toward large caps that rarely hit limit-up).")
        return
    lift = stats["lift"].to_numpy()
    qs = [50, 90, 99, 99.9, 100]
    print("lift percentiles (P(follower up | leader limit-up) - P(follower up | leader not)):")
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
    """Returns (raw_close, raw_preclose, suspend, adj_close, universe_stock_list)."""
    if args.source == "synthetic":
        raw_close, raw_preclose, suspend, adj_close, injected = lud.make_synthetic_limitup_market()
        print(f"[synthetic] injected {len(injected)} true limit-up-trigger pairs: {injected}")
        return raw_close, raw_preclose, suspend, adj_close, list(raw_close.columns)
    if args.source == "csv":
        if not (args.close_csv and args.preclose_csv):
            raise SystemExit("--close-csv and --preclose-csv are required for --source csv")
        raw_close, raw_preclose = ld.load_price_panels_csv(args.close_csv, args.preclose_csv)
        suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=raw_close.index, columns=raw_close.columns)
        adj_close = pd.read_csv(args.adjclose_csv, index_col=0, parse_dates=True) if args.adjclose_csv \
            else raw_close.copy()
        return raw_close, raw_preclose, suspend, adj_close, list(raw_close.columns)
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
        raw_close, raw_preclose, suspend, adj_close = lud.fetch_limitup_panels_xtdata(
            stock_list, start_time=args.start, end_time=args.end, download=not args.no_download
        )
        return raw_close, raw_preclose, suspend, adj_close, stock_list
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    raw_close, raw_preclose, suspend, adj_close, stock_list = load_panels(args)

    sector_map = None
    if not args.all_sectors:
        if args.source == "xtdata":
            sector_names = args.sector_names or ld.DEFAULT_SW_L1_SECTORS
            sector_map = ld.build_sector_map_xtdata(sector_names)
            mapped = sum(1 for c in stock_list if c in sector_map)
            print(f"sector map covers {mapped}/{len(stock_list)} stocks "
                  f"({len(set(sector_map.values()))} sectors used)")
        else:
            print("[limitup] --source is not xtdata: cannot build a real sector map, "
                  "so --all-sectors behavior is used regardless (pass real sector "
                  "boundaries via your own preprocessing if you need this restricted).")

    split = int(len(raw_close) * args.train_frac)

    def window(df, sl):
        return df.iloc[sl]

    train_sl, test_sl = slice(0, split), slice(split, None)
    print(f"train window: {raw_close.index[train_sl.start or 0]} .. {raw_close.index[split - 1]} "
          f"({split} days)")
    print(f"test window:  {raw_close.index[split]} .. {raw_close.index[-1]} "
          f"({len(raw_close) - split} days)")

    cfg = MiningConfig(lag=args.lag, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    leader_up_full, leader_valid_full = build_leader_frames(raw_close, raw_preclose, suspend, args.tolerance)
    returns_full = adj_close.sort_index().pct_change()
    follower_up_full, follower_valid_full = compute_up_indicator(returns_full, mode=args.mode, threshold=args.threshold)

    leader_up_train, leader_valid_train = window(leader_up_full, train_sl), window(leader_valid_full, train_sl)
    follower_up_train, follower_valid_train = window(follower_up_full, train_sl), window(follower_valid_full, train_sl)

    total_triggers = int(leader_up_train.to_numpy().sum())
    print(f"mining {leader_up_train.shape[1]} symbols x {leader_up_train.shape[0]} train days "
          f"({total_triggers} total limit-up days across the universe) "
          f"{'(same-sector pairs only)' if sector_map else ''}...")

    stats = compute_pairwise_stats_asymmetric(
        leader_up_train, leader_valid_train, follower_up_train, follower_valid_train, cfg, sector_map,
    )
    if not args.no_diagnostics:
        print_diagnostics(stats, cfg)

    if args.candidate_mode == "fdr":
        candidates = filter_significant_pairs(stats, cfg)
        print(f"{len(candidates)} candidate pairs survive FDR-controlled in-sample mining (alpha={cfg.alpha})")
    else:
        candidates = select_top_n_candidates(stats, cfg, args.top_n)
        print(f"{len(candidates)} candidate pairs taken by rank (top-n={args.top_n}, no FDR)")

    leader_up_test, leader_valid_test = window(leader_up_full, test_sl), window(leader_valid_full, test_sl)
    follower_up_test, follower_valid_test = window(follower_up_full, test_sl), window(follower_valid_full, test_sl)

    validated = validate_out_of_sample_asymmetric(
        leader_up_test, leader_valid_test, follower_up_test, follower_valid_test, candidates, cfg,
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
        "lag": args.lag,
        "tolerance": args.tolerance,
        "test_start": str(raw_close.index[split]) if split < len(raw_close) else None,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
