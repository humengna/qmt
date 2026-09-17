"""Grid-search leader_threshold x lag_bars for the T0-ETF pipeline
(research/run_etf_mining.py), fetching minute bars ONCE (xtdata/csv sources) and
reusing them across every combo in the grid - only the trigger detection and mining
stats differ per combo, not the underlying data, so there's no reason to re-download
or re-parse it per combo.

Why this exists: a single (leader_threshold=0.01, lag_bars=6) run over a longer, more
recent window found that the earlier "10 out-of-sample-significant pairs" result
(found on a shorter/older window) doesn't replicate at all (0/389 candidates passed
the out-of-sample FDR gate - see intraday/README.md's real-data section). Before
writing this hypothesis off, this sweeps a small grid of leader thresholds and
holding windows against ONE FIXED, EXPLICIT date range (pass --start/--end yourself -
comparing across combos is only meaningful if they all see the same data), with the
hub-follower filter (`leadlag.factor.exclude_hub_followers`) always applied, to check
whether some other configuration finds a relationship that's more than a narrow-
window/parameter-specific artifact.

`--top-liquid` is also worth trying alongside the grid: restricting to the most
liquid ETFs both reduces the "thin near-duplicate tracker" hub-follower risk and
makes the backtest's flat slippage assumption less unrealistic for the names that
survive.

Usage
-----
Smoke test on fabricated data (regenerates a fresh synthetic market per combo, since
the injected signal is calibrated to that combo's own threshold/lag_bars - this mode
checks the sweep mechanics, not a real parameter search)::

    python research/run_etf_param_sweep.py --source synthetic \\
        --thresholds 0.01 0.02 --lag-bars-list 3 6 --min-obs 15 --candidate-mode top-n --top-n 40

Real sweep against a running QMT/MiniQMT terminal, fixed date range::

    python research/run_etf_param_sweep.py --source xtdata --start 20200101 --end 20240601 \\
        --top-liquid 40 --thresholds 0.01 0.015 0.02 0.03 --lag-bars-list 3 6 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from intraday import data as idd
from intraday.event import build_follower_frames, build_leader_frames_threshold
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
from research.run_etf_mining import load_panels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data loading - same flag names as run_etf_mining.load_panels expects, so that
    # function can be reused here unchanged.
    p.add_argument("--source", choices=["xtdata", "csv", "synthetic"], default="synthetic")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--period", default="5m")
    p.add_argument("--close-csv")
    p.add_argument("--suspend-csv")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--top-liquid", type=int, default=None)
    # the grid
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.015, 0.02, 0.03])
    p.add_argument("--lag-bars-list", type=int, nargs="+", default=[3, 6, 12])
    # mining config, shared across every combo in the grid
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess")
    p.add_argument("--threshold", type=float, default=0.0, help="follower outcome threshold")
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
                    help="0 disables the hub-follower filter for this sweep")
    p.add_argument("--output-dir", default="research/output/etf_sweep")
    return p.parse_args()


def run_one_combo(minute_close, minute_suspend, leader_threshold, lag_bars, args):
    cfg = MiningConfig(lag=lag_bars, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    leader_triggered, leader_valid = build_leader_frames_threshold(
        minute_close, minute_suspend, threshold=leader_threshold
    )
    follower_outcome, follower_valid = build_follower_frames(
        minute_close, minute_suspend, lag_bars=lag_bars, mode=args.mode, threshold=args.threshold
    )

    split = int(len(minute_close) * args.train_frac)
    train_sl, test_sl = slice(0, split), slice(split, None)

    stats = compute_pairwise_stats_same_row(
        leader_triggered.iloc[train_sl], leader_valid.iloc[train_sl],
        follower_outcome.iloc[train_sl], follower_valid.iloc[train_sl], cfg, sector_map=None,
    )
    if args.candidate_mode == "fdr":
        candidates = filter_significant_pairs(stats, cfg)
    else:
        candidates = select_top_n_candidates(stats, cfg, args.top_n)

    validated = validate_out_of_sample_same_row(
        leader_triggered.iloc[test_sl], leader_valid.iloc[test_sl],
        follower_outcome.iloc[test_sl], follower_valid.iloc[test_sl], candidates, cfg,
    )
    oos_significant = filter_oos_significant(
        validated, alpha=args.oos_alpha, min_oos_lift=args.min_oos_lift, min_oos_n=args.min_oos_n
    )
    deduped = exclude_hub_followers(oos_significant, args.max_leaders_per_follower)
    deployable = select_for_deployment(deduped, max_unique_symbols=args.max_symbols, min_oos_n=args.min_oos_n)

    summary = {
        "leader_threshold": leader_threshold,
        "lag_bars": lag_bars,
        "n_train_triggers": int(leader_triggered.iloc[train_sl].to_numpy().sum()),
        "n_candidates": len(candidates),
        "n_oos_significant": len(oos_significant),
        "n_hub_excluded_pairs": len(oos_significant) - len(deduped),
        "n_deployable": len(deployable),
        "n_symbols": len(set(deployable["leader"]) | set(deployable["follower"])) if len(deployable) else 0,
        "mean_oos_z": float(deployable["oos_z"].mean()) if len(deployable) else float("nan"),
    }
    return summary, deployable, split


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_panels = None
    if args.source != "synthetic":
        minute_close, minute_suspend, stock_list = load_panels(args)
        shared_panels = (minute_close, minute_suspend)
        split_preview = int(len(minute_close) * args.train_frac)
        print(f"universe size: {len(stock_list)}")
        print(f"train window: {minute_close.index[0]} .. {minute_close.index[split_preview - 1]} "
              f"({split_preview} bars)")
        print(f"test window:  {minute_close.index[split_preview]} .. {minute_close.index[-1]} "
              f"({len(minute_close) - split_preview} bars)")
    else:
        print("[synthetic] regenerating a fresh synthetic market per combo (injected signal "
              "is calibrated to that combo's own threshold/lag_bars) - checks the sweep "
              "mechanics, not a real parameter search.")

    rows = []
    for leader_threshold in args.thresholds:
        for lag_bars in args.lag_bars_list:
            if shared_panels is not None:
                minute_close, minute_suspend = shared_panels
            else:
                minute_close, minute_suspend, _ = idd.make_synthetic_threshold_market(
                    lag_bars=lag_bars, threshold=leader_threshold
                )

            print(f"\n--- leader_threshold={leader_threshold:.1%} lag_bars={lag_bars} ---")
            summary, deployable, split = run_one_combo(minute_close, minute_suspend, leader_threshold, lag_bars, args)
            rows.append(summary)
            print(f"  train_triggers={summary['n_train_triggers']} candidates={summary['n_candidates']} "
                  f"oos_significant={summary['n_oos_significant']} hub_excluded_pairs={summary['n_hub_excluded_pairs']} "
                  f"deployable={summary['n_deployable']} symbols={summary['n_symbols']}")

            if len(deployable):
                fname = out_dir / f"pairs_thr{leader_threshold:g}_lag{lag_bars}.csv"
                deployable.to_csv(fname, index=False)
                fname.with_suffix(".meta.json").write_text(json.dumps({
                    "mode": args.mode, "threshold": args.threshold, "leader_threshold": leader_threshold,
                    "lag_bars": lag_bars, "period": args.period,
                    "test_start": str(minute_close.index[split]) if split < len(minute_close) else None,
                }, ensure_ascii=False, indent=2))

    summary_df = pd.DataFrame(rows).sort_values(["n_deployable", "mean_oos_z"], ascending=[False, False])
    summary_path = out_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n=== sweep summary (sorted by n_deployable, then mean_oos_z) ===")
    with pd.option_context("display.width", 140):
        print(summary_df.to_string(index=False))
    print(f"\nwrote {summary_path}; per-combo deployable pairs/meta under {out_dir}/ "
          f"(only written for combos with >= 1 deployable pair)")
    print("Remember: a combo showing up here still needs its OWN out-of-sample backtest "
          "(run_etf_backtest.py against its pairs/meta files) before trusting it - this "
          "sweep only re-applies the same in-sample/OOS-FDR/hub-filter gates already used "
          "everywhere else in this project, it does not replace the backtest step.")


if __name__ == "__main__":
    main()
