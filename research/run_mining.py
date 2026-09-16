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
train window (FDR-controlled, or rank-based via --candidate-mode=top-n) -> re-test
every candidate on the untouched test window -> keep only pairs that are themselves
FDR-significant OUT-OF-SAMPLE (checking just the SIGN of oos_lift is not enough - see
leadlag.factor.filter_oos_significant) -> trim to a symbol budget the live strategy
can actually subscribe to. See leadlag/factor.py and README.md for why every one of
these steps exists.
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
    compute_pairwise_stats,
    compute_returns,
    compute_up_indicator,
    filter_oos_significant,
    filter_significant_pairs,
    select_for_deployment,
    select_top_n_candidates,
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
    p.add_argument("--top-liquid", type=int, default=None,
                    help="(xtdata only) shrink the universe to the N stocks with the "
                         "highest median daily traded value before mining. Cuts O(N^2) "
                         "compute AND makes the FDR correction much less conservative "
                         "(fewer pairs tested -> lower bar to clear); see README.")
    p.add_argument("--no-download", action="store_true",
                    help="(xtdata only) skip download_history_data and read only the "
                         "already-cached local data; use this to re-run quickly with "
                         "different mining parameters after the first full download.")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess",
                    help="'excess' nets out each day's cross-sectional median return "
                         "before deciding 'up', to avoid mistaking common market moves "
                         "for a leader-follower relationship (recommended).")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--min-obs", type=int, default=60)
    p.add_argument("--candidate-mode", choices=["fdr", "top-n"], default="fdr",
                    help="'fdr' (default) keeps pairs surviving Benjamini-Hochberg at "
                         "--alpha; at whole-market scale (thousands to millions of pairs "
                         "tested at once) this routinely rejects everything even when "
                         "the strongest pairs have real lift, because FDR is calibrated "
                         "for the WHOLE tested family, not just the top few. 'top-n' "
                         "skips FDR and just takes the --top-n pairs by z-score, relying "
                         "on --min-lift plus the mandatory out-of-sample re-check "
                         "(validate_out_of_sample) to weed out false positives instead.")
    p.add_argument("--top-n", type=int, default=100,
                    help="candidate count when --candidate-mode=top-n")
    p.add_argument("--alpha", type=float, default=0.01, help="BH-FDR level for in-sample mining")
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--oos-alpha", type=float, default=0.05,
                    help="BH-FDR level applied to the OUT-OF-SAMPLE p-values. This is the "
                         "real gate - checking only whether oos_lift is positive lets "
                         "through ~50%% of pure noise by chance, regardless of how many "
                         "candidates were tested (this is not hypothetical: it's exactly "
                         "what a --candidate-mode=top-n or a loose --alpha run against the "
                         "whole market will otherwise do). Do not disable this.")
    p.add_argument("--min-oos-lift", type=float, default=0.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-symbols", type=int, default=500,
                    help="live-subscribe budget; see docs/QMT_API_NOTES.md")
    p.add_argument("--min-oos-n", type=int, default=20)
    p.add_argument("--output", default="research/output/pairs.csv")
    p.add_argument("--no-diagnostics", action="store_true",
                    help="skip printing the pre-filter lift/z distribution summary")
    p.add_argument("--same-sector-only", action="store_true",
                    help="(xtdata only) only test pairs whose leader and follower map to "
                         "the same industry/sector, instead of every ordered pair in the "
                         "universe. Uses --sector-names (or leadlag.data.DEFAULT_SW_L1_SECTORS "
                         "if not given) to build the mapping. Both targets a more plausible "
                         "hypothesis (co-movement within a sector) and shrinks the "
                         "multiple-testing family a lot.")
    p.add_argument("--sector-names", nargs="*", default=None,
                    help="sector/industry board names to group by (see --same-sector-only). "
                         "Must match names in YOUR client's sector tree - use --list-sectors "
                         "to check before assuming the default list works.")
    p.add_argument("--list-sectors", nargs="?", const="", default=None, metavar="NODE",
                    help="(xtdata only) print the sector-tree entries under NODE ('' = top "
                         "level) and exit, without running any mining. Use this to find the "
                         "real industry/concept board names in your client.")
    return p.parse_args()


def print_diagnostics(stats, cfg: MiningConfig) -> None:
    """Summarize the unfiltered lift/z distribution so a run that finds 0 pairs is
    distinguishable from 'genuinely no signal' vs 'signal exists but doesn't clear
    --min-lift/--alpha'. This is the single matrix computation `mine_lead_lag_pairs`
    would otherwise do internally - just inspected before the strict filtering.
    """
    n_pairs = len(stats)
    print(f"\n--- diagnostics: {n_pairs} pairs cleared --min-obs={cfg.min_obs} in each branch ---")
    if n_pairs == 0:
        print("no pair has >= min_obs observations in BOTH the leader-up and "
              "leader-not-up branches. Lower --min-obs, use a longer history, or "
              "check the universe actually has usable price history.")
        return

    lift = stats["lift"].to_numpy()
    z = stats["z"].to_numpy()
    qs = [50, 90, 99, 99.9, 100]
    lift_q = np.percentile(lift, qs)
    print("lift percentiles (P(follower up | leader up) - P(follower up | leader not up)):")
    for q, v in zip(qs, lift_q):
        print(f"  p{q:>5}: {v:+.4f}")
    print(f"pairs with lift >= --min-lift={cfg.min_lift}: {(lift >= cfg.min_lift).sum()} "
          f"(before FDR control)")

    top = stats.sort_values("z", ascending=False).head(10)
    print("top 10 pairs by z-score (NOT FDR-controlled - for inspection only):")
    with pd.option_context("display.width", 120):
        print(top[["leader", "follower", "n_leader_up", "p_cond", "p_base", "lift", "z"]]
              .to_string(index=False))
    print("--- end diagnostics ---\n")


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
        if args.top_liquid:
            stock_list = ld.rank_stocks_by_liquidity_xtdata(
                stock_list, start_time=args.start, end_time=args.end, top_n=args.top_liquid
            )
            print(f"kept top {len(stock_list)} by median daily traded value")
        return ld.fetch_price_panels_xtdata(
            stock_list, start_time=args.start, end_time=args.end, download=not args.no_download
        )
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()

    if args.list_sectors is not None:
        if args.source != "xtdata":
            raise SystemExit("--list-sectors requires --source xtdata")
        sector_names, folder_names = ld.list_sectors_xtdata(args.list_sectors)
        print(f"sectors under {args.list_sectors!r}: {sector_names}")
        print(f"folders under {args.list_sectors!r} (pass one as --list-sectors NODE "
              f"to look deeper): {folder_names}")
        return

    open_px, close_px = load_panels(args)

    sector_map = None
    if args.same_sector_only:
        if args.source != "xtdata":
            raise SystemExit("--same-sector-only requires --source xtdata")
        sector_names = args.sector_names or ld.DEFAULT_SW_L1_SECTORS
        sector_map = ld.build_sector_map_xtdata(sector_names)
        mapped = sum(1 for c in close_px.columns if c in sector_map)
        print(f"sector map covers {mapped}/{len(close_px.columns)} stocks in the universe "
              f"({len(set(sector_map.values()))} distinct sectors used)")
        if mapped < 0.5 * len(close_px.columns):
            print("WARNING: less than half the universe got a sector assignment - the "
                  "--sector-names probably don't match your client's actual sector-tree "
                  "names. Run --list-sectors to find the real ones.")

    returns = compute_returns(close_px)
    split = int(len(returns) * args.train_frac)
    train_returns, test_returns = returns.iloc[:split], returns.iloc[split:]
    print(f"train window: {train_returns.index.min()} .. {train_returns.index.max()} "
          f"({len(train_returns)} days)")
    print(f"test window:  {test_returns.index.min()} .. {test_returns.index.max()} "
          f"({len(test_returns)} days)")

    cfg = MiningConfig(lag=args.lag, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    up_train, valid_train = compute_up_indicator(train_returns, mode=args.mode, threshold=args.threshold)
    print(f"mining {up_train.shape[1]} symbols x {up_train.shape[0]} train days "
          f"{'(same-sector pairs only)' if sector_map else ''}...")
    stats = compute_pairwise_stats(up_train, valid_train, cfg, sector_map=sector_map)
    if not args.no_diagnostics:
        print_diagnostics(stats, cfg)

    if args.candidate_mode == "fdr":
        candidates = filter_significant_pairs(stats, cfg)
        print(f"{len(candidates)} candidate pairs survive FDR-controlled in-sample mining "
              f"(alpha={cfg.alpha})")
    else:
        candidates = select_top_n_candidates(stats, cfg, args.top_n)
        print(f"{len(candidates)} candidate pairs taken by rank (top-n={args.top_n}, no FDR - "
              f"out-of-sample validation below is the real filter)")

    up_test, valid_test = compute_up_indicator(test_returns, mode=args.mode, threshold=args.threshold)
    validated = validate_out_of_sample(up_test, valid_test, candidates, cfg)
    n_positive_sign = ((validated["oos_lift"] > 0) & (validated["oos_n"] >= args.min_oos_n)).sum()
    print(f"{n_positive_sign} pairs merely keep a positive oos_lift sign (NOT a real test - "
          f"~50% of pure noise passes this by chance; shown only for comparison)")

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
        "test_start": str(test_returns.index.min()) if len(test_returns) else None,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
