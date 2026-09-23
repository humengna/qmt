"""End-to-end mining pipeline for the SAME-DAY intraday spillover hypothesis.

Hypothesis under test: when a stock has a leader EVENT at some specific intraday bar,
other stocks in the SAME sector are more likely to rise over the following
`--lag-bars` bars, WITHIN THE SAME TRADING DAY, than their own baseline propensity.
`--trigger` picks what the event is:

  limitup (default) - the stock first touches its daily price limit (涨停). The size
      of the move is fixed by the board's rule, so it isn't a parameter.
  surge (急拉)       - the stock first rises `--leader-threshold` over the trailing
      `--surge-window` bars, i.e. both the SIZE and the SPEED of the move are
      parameters. A stock that grinds +3% up over four hours clears a since-open
      threshold but is not a surge, and does not trigger this one.

!!! T+1 WARNING: A-share equities cannot be sold on the day they were bought, so the
same-day round trip this pipeline's backtest models is NOT executable on stocks. The
mining is still a valid research question, and the intraday trigger is a
better-timed ENTRY than any daily signal, but a tradeable version has to hold to the
next day rather than exiting within the session. (T0-eligible instruments are a
different matter - see research/run_etf_mining.py, whose own hypothesis was tested
and rejected for unrelated reasons.)

This
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
from intraday.event import (
    build_follower_frames,
    build_follower_frames_overnight,
    build_leader_frames,
    build_leader_frames_surge,
)
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
    p.add_argument("--trigger", choices=["limitup", "surge"], default="limitup",
                    help="what counts as a leader event. 'limitup': first bar the stock "
                         "touches its daily price limit (size fixed by the board's rule). "
                         "'surge' (急拉): first bar its return over the trailing "
                         "--surge-window bars reaches --leader-threshold, i.e. both the "
                         "SIZE and the SPEED of the move are parameters. A stock that "
                         "grinds +3%% up over four hours is not a surge and does not "
                         "trigger - see intraday.data.compute_first_surge_indicator.")
    p.add_argument("--leader-threshold", type=float, default=0.02,
                    help="--trigger surge only: the rise that counts as a surge (default 2%%)")
    p.add_argument("--surge-window", type=int, default=5,
                    help="--trigger surge only: how many bars that rise must happen within "
                         "(default 5 bars; at --period 5m that is 25 minutes). The lookback "
                         "never crosses into the previous trading day, so an overnight gap "
                         "never counts as a surge.")
    p.add_argument("--tolerance", type=float, default=0.003, help="--trigger limitup only")
    p.add_argument("--include-st", action="store_true")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--hold", choices=["same-day", "overnight"], default="same-day",
                    help="what the follower's outcome - and therefore the tradeable hold - "
                         "is. 'same-day': did it rise over the next --lag-bars bars, within "
                         "the session (NOT executable on A-share equities, which are T+1). "
                         "'overnight': did it rise from the trigger bar to the NEXT trading "
                         "day's --exit-at bar, which is executable. Whichever you pick is "
                         "what gets statistically validated, and run_intraday_backtest.py "
                         "reads it back from meta.json so the backtest trades the same hold "
                         "the mining tested.")
    p.add_argument("--exit-at", choices=["next_open", "next_close"], default="next_open",
                    help="--hold overnight only: exit on the next trading day's first bar "
                         "(captures the overnight reaction alone) or its last bar (adds a "
                         "whole extra session of unrelated variance)")
    p.add_argument("--lag-bars", type=int, default=6,
                    help="--hold same-day only: how many bars ahead the follower's outcome "
                         "is measured (6 bars x 5m = 30 minutes); must stay within the day")
    p.add_argument("--min-obs", type=int, default=15,
                    help="intraday first-touch events are rarer still than daily "
                         "limit-up days, so this defaults even lower than limitup's 30")
    p.add_argument("--candidate-mode", choices=["fdr", "top-n"], default="fdr")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--oos-alpha", type=float, default=0.05)
    p.add_argument("--min-oos-lift", type=float, default=0.0)
    p.add_argument("--train-frac", type=float, default=0.7,
                    help="fraction of BARS used for training; ignored when --split-date is given")
    p.add_argument("--split-date", default=None,
                    help="split train/test at an explicit date (YYYYMMDD or YYYY-MM-DD) instead "
                         "of a bar fraction: bars before it train, bars from it on are the "
                         "held-out test window. Prefer this when you want a specific, "
                         "reproducible boundary - a fraction silently moves the boundary "
                         "whenever the fetched date range changes, which makes two runs "
                         "non-comparable without either of them looking wrong.")
    p.add_argument("--max-symbols", type=int, default=500)
    p.add_argument("--min-oos-n", type=int, default=10)
    p.add_argument("--max-leaders-per-follower", type=int, default=3,
                    help="drop any follower paired with more than this many distinct leaders "
                         "among the OOS-significant pairs, before the symbol-budget cap "
                         "(0 disables). A follower 'significant' against many unrelated "
                         "leaders at once is more likely high-beta to a shared factor than "
                         "genuinely driven by each of them - see "
                         "leadlag.factor.exclude_hub_followers. The share it drops is itself "
                         "a diagnostic: in the T0-ETF study it reached 97%%.")
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


def resolve_split(index: pd.DatetimeIndex, args) -> int:
    """Row index where the held-out test window starts: the first bar on or after
    --split-date, or --train-frac of the way through when no date is given.
    """
    if not args.split_date:
        return int(len(index) * args.train_frac)

    cutoff = pd.Timestamp(args.split_date)
    split = int(index.searchsorted(cutoff))
    if split <= 0 or split >= len(index):
        raise SystemExit(
            f"--split-date {args.split_date} leaves one side of the split empty: the fetched "
            f"bars run {index[0]} .. {index[-1]}. Pick a date inside that range, and remember "
            f"--start/--end decide what gets fetched in the first place."
        )
    return split


def build_leader_frames_for(args, minute_close, minute_suspend, daily_close):
    """Dispatch to the leader-trigger definition --trigger selects."""
    if args.trigger == "surge":
        return build_leader_frames_surge(
            minute_close, minute_suspend, threshold=args.leader_threshold, window_bars=args.surge_window
        )
    return build_leader_frames(minute_close, minute_suspend, daily_close, tolerance=args.tolerance)


def load_panels(args):
    """Returns (minute_close, minute_suspend, daily_close, stock_list). `daily_close` is
    None for --trigger surge, which needs no daily bars at all (only the limit-up trigger
    does, to derive each day's limit price) - and xtdata caches daily and minute bars
    separately, so not fetching what isn't needed also avoids depending on a daily cache
    that may not be populated.
    """
    if args.source == "synthetic":
        if args.trigger == "surge":
            minute_close, minute_suspend, injected = idd.make_synthetic_surge_market(
                lag_bars=args.lag_bars, threshold=args.leader_threshold, window_bars=args.surge_window
            )
            print(f"[synthetic] injected {len(injected)} true same-day surge pairs: {injected}")
            return minute_close, minute_suspend, None, list(minute_close.columns)
        minute_close, minute_suspend, daily_close, injected = idd.make_synthetic_intraday_market(
            lag_bars=args.lag_bars
        )
        print(f"[synthetic] injected {len(injected)} true same-day trigger pairs: {injected}")
        return minute_close, minute_suspend, daily_close, list(minute_close.columns)
    if args.source == "csv":
        if not args.close_csv:
            raise SystemExit("--close-csv is required for --source csv")
        if args.trigger == "limitup" and not args.daily_close_csv:
            raise SystemExit("--daily-close-csv is required for --source csv with --trigger limitup")
        minute_close = pd.read_csv(args.close_csv, index_col=0, parse_dates=True).sort_index()
        daily_close = pd.read_csv(args.daily_close_csv, index_col=0, parse_dates=True).sort_index() \
            if args.daily_close_csv else None
        minute_suspend = pd.read_csv(args.suspend_csv, index_col=0, parse_dates=True) if args.suspend_csv \
            else pd.DataFrame(0, index=minute_close.index, columns=minute_close.columns)
        return minute_close, minute_suspend, daily_close, list(minute_close.columns)
    if args.source == "xtdata":
        stock_list = ld.get_full_market_stock_list_xtdata(tuple(args.sectors))
        print(f"universe size: {len(stock_list)}")
        if not stock_list:
            raise SystemExit(
                f"--sectors {' '.join(args.sectors)} matched no stocks. get_stock_list_in_sector "
                f"returns an EMPTY LIST for a board name this QMT client doesn't have rather "
                f"than raising, and board naming differs between installations and data vendors "
                f"(one client's 'SW1电子' may be '电子' or absent in another). Dump the names this "
                f"client actually has:\n"
                f"    python -c \"import sys; sys.path.insert(0,'.'); "
                f"from leadlag.data import list_sectors_xtdata; "
                f"names, _ = list_sectors_xtdata(); print(len(names)); print([n for n in names if '电子' in n])\""
            )
        if not args.include_st:
            st_codes = lud.build_st_exclusion_set_xtdata(stock_list)
            stock_list = [c for c in stock_list if c not in st_codes]
            print(f"excluded {len(st_codes)} ST/*ST names, {len(stock_list)} remain")
        if args.top_liquid:
            stock_list = ld.rank_stocks_by_liquidity_xtdata(
                stock_list, start_time=args.start, end_time=args.end, top_n=args.top_liquid
            )
            print(f"kept top {len(stock_list)} by median daily traded value")
        if args.trigger == "surge":
            minute_close, minute_suspend = idd.fetch_intraday_close_panels_xtdata(
                stock_list, start_time=args.start, end_time=args.end, period=args.period,
                download=not args.no_download,
            )
            daily_close = None
        else:
            minute_close, minute_suspend, daily_close = idd.fetch_intraday_panels_xtdata(
                stock_list, start_time=args.start, end_time=args.end, period=args.period,
                download=not args.no_download,
            )
        if minute_close.empty or minute_close.shape[1] == 0:
            raise SystemExit(
                f"no {args.period} bars returned for {len(stock_list)} symbol(s) over "
                f"{args.start or '(open)'}..{args.end or '(open)'}. Check the date range covers "
                f"real trading days and that this period's history is downloaded locally "
                f"(drop --no-download, or use QMT's 数据管理)."
            )
        if args.trigger == "limitup" and (daily_close is None or daily_close.empty
                                          or not daily_close.notna().to_numpy().any()):
            raise SystemExit(
                "--trigger limitup needs DAILY bars to derive each day's limit price from the "
                "prior close, and none came back. xtdata caches 1d separately from "
                f"{args.period}, so having minute history does NOT mean the daily history is "
                "there. Without this the limit price is NaN, every bar is marked invalid, and "
                "the run would just report 0 trigger events. Check the daily cache with\n"
                f"    python research/check_data_coverage.py --period 1d --start {args.start or '20230103'}\n"
                "then drop --no-download to fetch it, or use --trigger surge, which needs no "
                "daily bars at all."
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

    split = resolve_split(minute_close.index, args)
    train_sl, test_sl = slice(0, split), slice(split, None)
    print(f"train window: {minute_close.index[0]} .. {minute_close.index[split - 1]} ({split} bars)")
    print(f"test window:  {minute_close.index[split]} .. {minute_close.index[-1]} "
          f"({len(minute_close) - split} bars)")

    cfg = MiningConfig(lag=args.lag_bars, min_obs=args.min_obs, alpha=args.alpha, min_lift=args.min_lift)

    leader_triggered_full, leader_valid_full = build_leader_frames_for(
        args, minute_close, minute_suspend, daily_close
    )
    if args.hold == "overnight":
        follower_outcome_full, follower_valid_full = build_follower_frames_overnight(
            minute_close, minute_suspend, exit_at=args.exit_at, mode=args.mode, threshold=args.threshold
        )
    else:
        follower_outcome_full, follower_valid_full = build_follower_frames(
            minute_close, minute_suspend, lag_bars=args.lag_bars, mode=args.mode, threshold=args.threshold
        )

    total_triggers = int(leader_triggered_full.iloc[train_sl].to_numpy().sum())
    event_label = (f"+{args.leader_threshold:.1%}-in-{args.surge_window}-bar surge"
                   if args.trigger == "surge" else "first-touch-limit")
    hold_label = (f"held to the next day's {args.exit_at}" if args.hold == "overnight"
                  else f"held {args.lag_bars} bars, same session (T+1: NOT executable on stocks)")
    print(f"mining {leader_triggered_full.shape[1]} symbols x {split} train bars "
          f"({total_triggers} total {event_label} events across the universe) "
          f"{'(same-sector pairs only)' if sector_map else ''}; follower outcome = {hold_label}...")

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

    deduped = exclude_hub_followers(oos_significant, args.max_leaders_per_follower)
    n_hub_followers = len(set(oos_significant["follower"]) - set(deduped["follower"])) if len(oos_significant) else 0
    if n_hub_followers:
        print(f"excluded {n_hub_followers} 'hub' follower(s) paired with more than "
              f"--max-leaders-per-follower={args.max_leaders_per_follower} distinct leaders "
              f"({len(oos_significant) - len(deduped)} pairs dropped) - likely shared-factor/"
              f"beta exposure, not a real per-leader relationship")

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
        "lag_bars": args.lag_bars,
        "hold": args.hold,
        "exit_at": args.exit_at,
        "trigger": args.trigger,
        "leader_threshold": args.leader_threshold,
        "surge_window": args.surge_window,
        "tolerance": args.tolerance,
        "period": args.period,
        "test_start": str(minute_close.index[split]) if split < len(minute_close) else None,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
