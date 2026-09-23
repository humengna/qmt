"""Batch-run run_intraday_mining.py once per sector (default: all 31 SW1 industries).

An intraday (minute-bar) mining pass over the "whole market" for a multi-year date
range does not fit in memory on ordinary hardware in one shot (see
intraday/README.md's data-volume warning): ~5000 stocks x ~48 five-minute bars/day x
~1600 trading days is roughly 380 million bar-records, tens of GB just for the close
price field. Because --same-sector-only means pairs never cross sector boundaries
anyway, running one sector at a time loses NO statistical coverage versus a single
whole-market run - it just does the same work as ~31 small, memory-bounded jobs
instead of one impossible one, and a sector that fails or hangs can be identified and
retried without redoing the others.

Usage::

    python research/run_intraday_mining_by_sector.py --start 20200101 \\
        --candidate-mode top-n --top-n 30

Each sector writes research/output/by_sector/intraday_pairs_<sector>.csv (+
.meta.json) and a log to research/output/by_sector/logs/<sector>.log. All sectors'
pairs are concatenated, and the LIVE-SUBSCRIBE SYMBOL BUDGET (default 500, same as
leadlag/limitup) is re-applied GLOBALLY across the merged set - each sector's own run
also caps at 500, but that is a per-sector formality; merging 31 sectors can easily
exceed 500 unique symbols in total, and that combined file is what you'd actually feed
to one live QMT strategy instance. Re-run with --skip-existing to resume after an
interruption without redoing finished sectors.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from leadlag.data import DEFAULT_SW_L1_SECTORS
from leadlag.factor import select_for_deployment

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
_MINING_SCRIPT = Path(__file__).resolve().parent / "run_intraday_mining.py"


def _safe_filename(name: str) -> str:
    return _INVALID_FILENAME_CHARS.sub("_", name)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sectors", nargs="*", default=None,
                    help="which sectors to loop over (default: all 31 SW1 industries)")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--period", default="5m")
    p.add_argument("--lag-bars", type=int, default=6)
    p.add_argument("--min-obs", type=int, default=15)
    p.add_argument("--candidate-mode", choices=["fdr", "top-n"], default="top-n")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--min-lift", type=float, default=0.05)
    p.add_argument("--oos-alpha", type=float, default=0.1)
    p.add_argument("--train-frac", type=float, default=0.7,
                    help="ignored when --split-date is given")
    p.add_argument("--split-date", default=None,
                    help="explicit train/test boundary (YYYYMMDD), passed through to each sector's run - see run_intraday_mining.py --split-date")
    p.add_argument("--min-oos-n", type=int, default=10)
    p.add_argument("--tolerance", type=float, default=0.003, help="--trigger limitup only")
    p.add_argument("--trigger", choices=["limitup", "surge"], default="limitup",
                    help="leader event definition, passed through to the per-sector mining "
                         "run - see run_intraday_mining.py's --trigger")
    p.add_argument("--leader-threshold", type=float, default=0.02, help="--trigger surge only")
    p.add_argument("--surge-window", type=int, default=5, help="--trigger surge only")
    p.add_argument("--hold", choices=["same-day", "overnight"], default="same-day",
                    help="follower outcome / tradeable hold, passed through - 'overnight' is "
                         "the T+1-executable one for stocks. See run_intraday_mining.py --hold")
    p.add_argument("--exit-at", choices=["next_open", "next_close"], default="next_open",
                    help="--hold overnight only")
    p.add_argument("--max-leaders-per-follower", type=int, default=3,
                    help="hub-follower filter applied WITHIN each sector's run; the merge step "
                         "below re-applies only the global symbol-budget cap, so a follower "
                         "that looks fine inside its own sector is not re-checked across "
                         "sectors (with --same-sector-only there are no cross-sector pairs "
                         "anyway, so per-sector is the whole family)")
    p.add_argument("--max-symbols", type=int, default=500,
                    help="GLOBAL live-subscribe budget applied once at merge time")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--include-st", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                    help="skip a sector whose output CSV already exists (resume after an interruption)")
    p.add_argument("--output-dir", default="research/output/by_sector")
    p.add_argument("--merged-output", default="research/output/intraday_pairs_all.csv")
    p.add_argument("--timeout", type=int, default=3600,
                    help="per-sector subprocess timeout in seconds (default 1h). A "
                         "sector that hangs past this is killed and marked failed; "
                         "large sectors or long date ranges may need more.")
    return p.parse_args()


def run_one_sector(sector: str, args, out_dir: Path, log_dir: Path) -> tuple[str, bool, int]:
    safe = _safe_filename(sector)
    output_csv = out_dir / f"intraday_pairs_{safe}.csv"
    log_path = log_dir / f"{safe}.log"

    if args.skip_existing and output_csv.exists():
        n = len(pd.read_csv(output_csv)) if output_csv.stat().st_size > 0 else 0
        print(f"[{sector}] output already exists, skipping (--skip-existing): {n} pairs")
        return sector, True, n

    cmd = [
        sys.executable, str(_MINING_SCRIPT),
        "--source", "xtdata",
        "--sectors", sector,
        "--sector-names", sector,
        "--start", args.start,
        "--end", args.end,
        "--period", args.period,
        "--lag-bars", str(args.lag_bars),
        "--min-obs", str(args.min_obs),
        "--candidate-mode", args.candidate_mode,
        "--top-n", str(args.top_n),
        "--alpha", str(args.alpha),
        "--min-lift", str(args.min_lift),
        "--oos-alpha", str(args.oos_alpha),
        "--train-frac", str(args.train_frac),
        *(["--split-date", args.split_date] if args.split_date else []),
        "--min-oos-n", str(args.min_oos_n),
        "--tolerance", str(args.tolerance),
        "--trigger", args.trigger,
        "--leader-threshold", str(args.leader_threshold),
        "--surge-window", str(args.surge_window),
        "--hold", args.hold,
        "--exit-at", args.exit_at,
        "--max-leaders-per-follower", str(args.max_leaders_per_follower),
        "--output", str(output_csv),
    ]
    if args.no_download:
        cmd.append("--no-download")
    if args.include_st:
        cmd.append("--include-st")

    print(f"\n===== [{sector}] starting (log: {log_path}) =====")
    with open(log_path, "w", encoding="utf-8") as log_f:
        try:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[{sector}] TIMED OUT after {args.timeout}s - see {log_path}")
            return sector, False, 0

    if proc.returncode != 0:
        print(f"[{sector}] FAILED (exit code {proc.returncode}) - see {log_path}")
        return sector, False, 0

    n = len(pd.read_csv(output_csv)) if output_csv.exists() and output_csv.stat().st_size > 0 else 0
    print(f"[{sector}] done: {n} pairs")
    return sector, True, n


def main():
    args = parse_args()
    sectors = args.sectors or DEFAULT_SW_L1_SECTORS

    out_dir = Path(args.output_dir)
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = [run_one_sector(sector, args, out_dir, log_dir) for sector in sectors]

    print("\n===== summary =====")
    frames = []
    for sector, ok, n in results:
        print(f"{'OK' if ok else 'FAILED':8s} {sector:12s} {n} pairs")
        if ok:
            csv_path = out_dir / f"intraday_pairs_{_safe_filename(sector)}.csv"
            if csv_path.exists() and csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    frames.append(df)

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(sectors)} sectors completed successfully")
    failed = [s for s, ok, _ in results if not ok]
    if failed:
        print(f"failed sectors (re-run with --sectors {' '.join(failed)} --skip-existing "
              f"to retry just these, everything else stays cached): {failed}")

    if not frames:
        print("no sector produced any deployable pairs - nothing to merge")
        return

    merged_raw = pd.concat(frames, ignore_index=True)
    n_raw_symbols = len(set(merged_raw["leader"]) | set(merged_raw["follower"]))
    print(f"{len(merged_raw)} total pairs across all sectors before the global "
          f"live-subscribe budget cap ({n_raw_symbols} unique symbols)")

    merged = select_for_deployment(
        merged_raw, max_unique_symbols=args.max_symbols, min_oos_n=args.min_oos_n
    )
    n_symbols = len(set(merged["leader"]) | set(merged["follower"])) if len(merged) else 0
    print(f"{len(merged)} pairs kept after the global budget cap ({n_symbols} unique "
          f"symbols, budget={args.max_symbols})")

    merged_path = Path(args.merged_output)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)

    # meta.json is identical across sectors (every subprocess call used the same
    # mode/threshold/lag_bars/tolerance/period) - copy any one successful sector's.
    for sector, ok, n in results:
        if ok:
            meta_src = out_dir / f"intraday_pairs_{_safe_filename(sector)}.meta.json"
            if meta_src.exists():
                merged_path.with_suffix(".meta.json").write_text(
                    meta_src.read_text(encoding="utf-8"), encoding="utf-8"
                )
                break

    print(f"wrote {merged_path} and {merged_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
