"""End-to-end mining pipeline for the T0-ETF intraday spillover hypothesis.

Hypothesis under test: within a FIXED universe of T0-tradable ETFs (cross-border
HK/US, commodity, bond - see SYMBOL_LIST below), when one ETF's cumulative return
since the day's own first bar FIRST crosses `--leader-threshold` (default 1%) at some
intraday bar, another ETF in the SAME universe is more likely to be up over the
following `--lag-bars` bars, WITHIN THE SAME TRADING DAY, than its own baseline
propensity.

This differs from run_intraday_mining.py in exactly the ways the user asked for:
  - no sector split at all - every ordered pair within SYMBOL_LIST is tested directly
    (sector_map=None always), since the whole point of a T0 ETF universe is that these
    instruments don't share a 申万一级 industry structure the way stocks do.
  - the leader trigger is a plain "crossed 1% since today's open" threshold
    (`intraday.event.build_leader_frames_threshold`), not a 涨停/price-limit formula -
    most of these ETFs never seal at a board limit at all, so run_intraday_mining.py's
    涨停-based `build_leader_frames` would find ~zero leader events here.
Everything else - the same-row-aligned mining/OOS-validation/FDR machinery, the
follower's forward-return outcome, the deployment symbol-budget cap - is reused
unchanged from leadlag.factor / intraday.event.build_follower_frames.

Usage examples
--------------
Smoke test on fabricated data (no QMT / network needed)::

    python research/run_etf_mining.py --source synthetic --min-obs 15 \\
        --candidate-mode top-n --top-n 40 --output research/output/etf_pairs.csv

Mine on real 5-minute bars from a running QMT/MiniQMT terminal::

    python research/run_etf_mining.py --source xtdata --start 20220101 --end 20240601 \\
        --output research/output/etf_pairs.csv

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
from intraday.event import build_follower_frames, build_leader_frames_threshold
from leadlag import data as ld
from leadlag.factor import (
    MiningConfig,
    compute_pairwise_stats_same_row,
    exclude_hub_followers,
    filter_oos_significant,
    filter_significant_pairs,
    select_for_deployment,
    select_top_n_candidates,
    validate_out_of_sample_same_row,
)

# User-confirmed T0-tradable ETF universe (cross-border HK/US, commodity, bond).
# T0 eligibility is a real-world settlement-rule fact this project has no way to verify
# programmatically (see docs/QMT_API_NOTES.md) - this list was supplied by the user,
# not derived here.
SYMBOL_LIST = [
    # === 跨境ETF — 港股 ===
    '513050.SH', '513130.SH', '513330.SH', '513580.SH', '513890.SH',
    '513010.SH', '513260.SH', '513380.SH', '513690.SH', '513180.SH',
    '513660.SH', '513680.SH', '513060.SH', '513320.SH', '513600.SH',
    '513960.SH', '513530.SH', '513550.SH', '513990.SH', '513280.SH',
    '513770.SH', '513860.SH', '513980.SH', '513020.SH', '513160.SH',
    '513230.SH', '513150.SH', '513070.SH', '513120.SH', '513140.SH',
    '513900.SH', '513090.SH', '513700.SH', '513390.SH',
    '159607.SZ', '159605.SZ', '159742.SZ', '159740.SZ', '159741.SZ',
    '159850.SZ', '159726.SZ', '159892.SZ', '159920.SZ', '159712.SZ',
    '159735.SZ', '159792.SZ', '159636.SZ', '159751.SZ', '159788.SZ',
    '159823.SZ', '159954.SZ', '159960.SZ', '159750.SZ', '159747.SZ',
    '159711.SZ', '159615.SZ',
    '510900.SH',   # H股ETF
    # === 跨境ETF — 美股/其他 ===
    '513300.SH', '513100.SH', '513500.SH', '513520.SH', '513030.SH',
    '513080.SH', '513220.SH', '513360.SH',
    '159941.SZ', '159655.SZ', '159866.SZ', '159822.SZ',
    # === 商品ETF ===
    '518880.SH', '518800.SH', '518660.SH', '518850.SH', '518860.SH',
    '159937.SZ', '159934.SZ', '159833.SZ', '159981.SZ', '159980.SZ',
    '159985.SZ',
    # === 债券ETF ===
    '511010.SH', '511260.SH', '511380.SH', '511360.SH', '511520.SH',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--period", default="5m", help="bar granularity for get_market_data_ex/download_history_data")
    p.add_argument("--close-csv", help="wide CSV of minute close prices (xtdata alternative)")
    p.add_argument("--suspend-csv")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--top-liquid", type=int, default=None,
                    help="(xtdata only) shrink SYMBOL_LIST to the N most liquid ETFs by "
                         "median daily traded value before mining. Also a hedge against "
                         "the 'hub follower' pattern (see exclude_hub_followers): the "
                         "least liquid names in a fixed cross-border/commodity ETF list "
                         "are the most likely to be thin near-duplicate trackers whose "
                         "'up' co-movement with everything else is a liquidity/NAV-lag "
                         "artifact rather than a real relationship, and thin names are "
                         "also where the backtest's flat slippage assumption is least "
                         "realistic (likely understating true cost).")
    p.add_argument("--leader-threshold", type=float, default=0.01,
                    help="leader trigger: cumulative return since the day's first bar "
                         "first reaches this (default 0.01 = 1%%). No limit-up formula "
                         "involved - see intraday/data.py's "
                         "compute_first_threshold_cross_indicator.")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess",
                    help="follower outcome definition (excess nets out the cross-"
                         "sectional median forward return across the ETF universe)")
    p.add_argument("--threshold", type=float, default=0.0, help="follower outcome threshold")
    p.add_argument("--lag-bars", type=int, default=6,
                    help="how many bars ahead the follower's outcome is measured "
                         "(6 bars x 5m = 30 minutes); must stay within the same trading day")
    p.add_argument("--min-obs", type=int, default=15)
    p.add_argument("--candidate-mode", choices=["fdr", "top-n"], default="fdr")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--oos-alpha", type=float, default=0.05)
    p.add_argument("--min-oos-lift", type=float, default=0.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-symbols", type=int, default=500)
    p.add_argument("--min-oos-n", type=int, default=10)
    p.add_argument("--max-leaders-per-follower", type=int, default=3,
                    help="drop any follower paired with more than this many distinct "
                         "leaders among the OOS-significant pairs, before the symbol-"
                         "budget cap (0 disables). This ETF universe is small and highly "
                         "homogeneous (85 cross-border/commodity/bond names), so a "
                         "follower 'significant' against a dozen+ unrelated leaders at "
                         "once is much more likely to be a high-beta name riding a "
                         "shared factor excess-mode didn't fully net out, than a dozen+ "
                         "genuine per-leader relationships - see "
                         "leadlag.factor.exclude_hub_followers's docstring.")
    p.add_argument("--output", default="research/output/etf_pairs.csv")
    p.add_argument("--no-diagnostics", action="store_true")
    return p.parse_args()


def print_diagnostics(stats: pd.DataFrame, cfg: MiningConfig) -> None:
    n_pairs = len(stats)
    print(f"\n--- diagnostics: {n_pairs} pairs cleared --min-obs={cfg.min_obs} in each branch ---")
    if n_pairs == 0:
        print("no pair has enough threshold-cross bars in both branches. Lower "
              "--leader-threshold, lower --min-obs, or widen the date range.")
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
    """Returns (minute_close, minute_suspend, stock_list)."""
    if args.source == "synthetic":
        minute_close, minute_suspend, injected = idd.make_synthetic_threshold_market(
            lag_bars=args.lag_bars, threshold=args.leader_threshold,
        )
        print(f"[synthetic] injected {len(injected)} true same-day trigger pairs: {injected}")
        return minute_close, minute_suspend, list(minute_close.columns)
    if args.source == "csv":
        if not args.close_csv:
            raise SystemExit("--close-csv is required for --source csv")
        minute_close = pd.read_csv(args.close_csv, index_col=0, parse_dates=True).sort_index()
        minute_suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=minute_close.index, columns=minute_close.columns)
        return minute_close, minute_suspend, list(minute_close.columns)
    if args.source == "xtdata":
        universe = SYMBOL_LIST
        if args.top_liquid:
            universe = ld.rank_stocks_by_liquidity_xtdata(
                universe, start_time=args.start, end_time=args.end, top_n=args.top_liquid
            )
            print(f"kept top {len(universe)} of {len(SYMBOL_LIST)} ETFs by median daily traded value")
        minute_close, minute_suspend, _daily_close = idd.fetch_intraday_panels_xtdata(
            universe, start_time=args.start, end_time=args.end, period=args.period,
            download=not args.no_download,
        )
        return minute_close, minute_suspend, universe
    raise SystemExit(f"unknown source {args.source}")


def main():
    args = parse_args()
    minute_close, minute_suspend, stock_list = load_panels(args)
    print(f"universe size: {len(stock_list)} (fixed T0 ETF list, no sector split)")

    split = int(len(minute_close) * args.train_frac)
    train_sl, test_sl = slice(0, split), slice(split, None)
    print(f"train window: {minute_close.index[0]} .. {minute_close.index[split - 1]} ({split} bars)")
    print(f"test window:  {minute_close.index[split]} .. {minute_close.index[-1]} "
          f"({len(minute_close) - split} bars)")

    cfg = MiningConfig(lag=args.lag_bars, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    leader_triggered_full, leader_valid_full = build_leader_frames_threshold(
        minute_close, minute_suspend, threshold=args.leader_threshold
    )
    follower_outcome_full, follower_valid_full = build_follower_frames(
        minute_close, minute_suspend, lag_bars=args.lag_bars, mode=args.mode, threshold=args.threshold
    )

    total_triggers = int(leader_triggered_full.iloc[train_sl].to_numpy().sum())
    print(f"mining {leader_triggered_full.shape[1]} symbols x {split} train bars "
          f"({total_triggers} total >= {args.leader_threshold:.1%} intraday-rise events "
          f"across the universe, no sector restriction)...")

    stats = compute_pairwise_stats_same_row(
        leader_triggered_full.iloc[train_sl], leader_valid_full.iloc[train_sl],
        follower_outcome_full.iloc[train_sl], follower_valid_full.iloc[train_sl], cfg, sector_map=None,
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

    deduped = exclude_hub_followers(oos_significant, args.max_leaders_per_follower)
    n_hub_followers = len(set(oos_significant["follower"]) - set(deduped["follower"])) if len(oos_significant) else 0
    if n_hub_followers:
        print(f"excluded {n_hub_followers} 'hub' follower(s) paired with more than "
              f"--max-leaders-per-follower={args.max_leaders_per_follower} distinct "
              f"leaders ({len(oos_significant) - len(deduped)} pairs dropped) - likely "
              f"shared-factor/beta exposure, not a real per-leader relationship")

    deployable = select_for_deployment(
        deduped, max_unique_symbols=args.max_symbols, min_oos_n=args.min_oos_n
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
        "leader_threshold": args.leader_threshold,
        "lag_bars": args.lag_bars,
        "period": args.period,
        "test_start": str(minute_close.index[split]) if split < len(minute_close) else None,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
