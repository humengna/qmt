"""Diagnose why specific stocks did or didn't end up in a mined pairs table.

Answers the most common version of "why isn't stock X in my results":

  1. Does X appear anywhere in the merged (post-global-cap) pairs table?
  2. Does X appear in any PER-SECTOR pairs table (i.e. it was a real out-of-sample
     -significant candidate, but got cut later by the global 500-symbol live-
     subscribe budget - see run_intraday_mining_by_sector.py)?
  3. (--source xtdata) Is X even usable in the first place: is it ST-excluded, does
     it map to one of the default 31 SW1 sectors, and - almost always the real
     answer - how many raw trigger events (daily "up" days / limit-up days / intraday
     first-touches, depending on --kind) does it actually have in the date range?
     The event every one of these three pipelines looks for is rare; a stock that
     never clears --min-obs as a LEADER can never produce a candidate pair no matter
     how strong any real relationship might be, and this checks that directly instead
     of guessing.

A stock can still appear as a FOLLOWER even with zero trigger events of its own
(follower status doesn't require the code to ever trigger anything itself) - this
only diagnoses its viability AS A LEADER, which is the far more common reason
someone expected a specific stock to show up and it didn't.

Usage examples
--------------
Check where a set of codes ended up (or didn't) in existing output files, no network
needed::

    python research/diagnose_stocks.py --codes 000657 600549 002842 002378 \\
        --pairs research/output/intraday_pairs_all.csv \\
        --by-sector-dir research/output/by_sector

Add a live check of ST status / sector / raw trigger-event counts (requires a running
QMT/MiniQMT terminal)::

    python research/diagnose_stocks.py --codes 000657 600549 002842 002378 \\
        --source xtdata --kind intraday --start 20200101 --period 5m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def normalize_code(code: str) -> str:
    code = code.strip()
    if "." in code:
        return code.upper()
    prefix2 = code[:2]
    if prefix2 in ("60", "68", "90"):
        return f"{code}.SH"
    if prefix2 in ("00", "30", "20"):
        return f"{code}.SZ"
    if code[:1] in ("8", "4"):
        return f"{code}.BJ"
    raise ValueError(f"can't infer the market suffix for {code!r}; pass it explicitly, e.g. '000657.SZ'")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes", nargs="+", required=True,
                    help="stock codes, with or without market suffix (e.g. 000657 or 000657.SZ)")
    p.add_argument("--pairs", help="a merged/final pairs.csv to check (e.g. intraday_pairs_all.csv)")
    p.add_argument("--by-sector-dir", help="a directory of per-sector pairs CSVs (pre-global-cap)")
    p.add_argument("--source", choices=["xtdata"], default=None,
                    help="add a live check (ST status, sector, raw trigger-event count)")
    p.add_argument("--kind", choices=["leadlag", "limitup", "intraday"], default="intraday",
                    help="which pipeline's leader-trigger definition to check live")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--period", default="5m", help="intraday kind only")
    p.add_argument("--mode", choices=["absolute", "excess"], default="excess", help="leadlag kind only")
    p.add_argument("--threshold", type=float, default=0.0, help="leadlag kind only")
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def check_pairs_file(path: Path, codes: list[str], label: str) -> None:
    if not path.exists():
        print(f"  ({label}: file not found: {path})")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    for code in codes:
        as_leader = df[df["leader"] == code]
        as_follower = df[df["follower"] == code]
        if as_leader.empty and as_follower.empty:
            continue
        print(f"  [{label}] {code} found in {path.name}:")
        if not as_leader.empty:
            print(f"    as LEADER   -> followers: {as_leader['follower'].tolist()}")
        if not as_follower.empty:
            print(f"    as FOLLOWER <- leaders:   {as_follower['leader'].tolist()}")


def scan_by_sector_dir(by_sector_dir: str, codes: list[str]) -> None:
    d = Path(by_sector_dir)
    if not d.exists():
        print(f"  (--by-sector-dir not found: {d})")
        return
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        print(f"  (no per-sector CSVs found under {d})")
        return
    any_found = False
    for csv_path in csvs:
        before = any_found
        df = pd.read_csv(csv_path) if csv_path.stat().st_size > 0 else pd.DataFrame()
        if not df.empty and any(c in set(df["leader"]) | set(df["follower"]) for c in codes):
            any_found = True
        check_pairs_file(csv_path, codes, label=csv_path.stem)
    if not any_found:
        print("  none of the codes appear in ANY per-sector output - they never "
              "became out-of-sample-significant candidates anywhere (see the live "
              "--source xtdata check below for why)")


def live_checks(codes: list[str], args) -> None:
    from leadlag.data import DEFAULT_SW_L1_SECTORS, build_sector_map_xtdata

    print("\n--- live xtdata checks ---")
    try:
        from limitup.data import build_st_exclusion_set_xtdata
        st_codes = build_st_exclusion_set_xtdata(codes)
    except Exception as exc:  # noqa: BLE001 - ST check is best-effort, don't block the rest
        print(f"(ST check unavailable: {exc})")
        st_codes = set()

    sector_map = build_sector_map_xtdata(DEFAULT_SW_L1_SECTORS)

    for code in codes:
        print(f"\n{code}:")
        print(f"  currently ST-excluded by default (--include-st to override): {code in st_codes}")
        print(f"  maps to sector: {sector_map.get(code, '(none of the default 31 SW1 sectors - see --list-sectors)')}")

        try:
            if args.kind == "limitup":
                from limitup.data import fetch_limitup_panels_xtdata
                from limitup.event import build_leader_frames
                raw_close, raw_preclose, suspend, _ = fetch_limitup_panels_xtdata(
                    [code], start_time=args.start, end_time=args.end, download=not args.no_download,
                )
                if code not in raw_close.columns or raw_close[code].dropna().empty:
                    print("  no daily price history returned for this code/date range")
                    continue
                triggered, valid = build_leader_frames(raw_close, raw_preclose, suspend)
                print(f"  valid trading days in range: {int(valid[code].sum())}")
                print(f"  limit-up days (usable AS A LEADER): {int(triggered[code].sum())}")
            elif args.kind == "intraday":
                from intraday.data import fetch_intraday_panels_xtdata
                from intraday.event import build_leader_frames
                minute_close, minute_suspend, daily_close = fetch_intraday_panels_xtdata(
                    [code], start_time=args.start, end_time=args.end, period=args.period,
                    download=not args.no_download,
                )
                if code not in minute_close.columns or minute_close[code].dropna().empty:
                    print("  no minute price history returned for this code/date range")
                    continue
                triggered, valid = build_leader_frames(minute_close, minute_suspend, daily_close)
                print(f"  valid bars in range: {int(valid[code].sum())}")
                print(f"  first-touch-limit events (usable AS A LEADER): {int(triggered[code].sum())}")
            else:
                from leadlag.data import fetch_price_panels_xtdata
                from leadlag.factor import compute_returns, compute_up_indicator
                open_px, close_px = fetch_price_panels_xtdata(
                    [code], start_time=args.start, end_time=args.end, download=not args.no_download,
                )
                if code not in close_px.columns or close_px[code].dropna().empty:
                    print("  no daily price history returned for this code/date range")
                    continue
                returns = compute_returns(close_px)
                up, valid = compute_up_indicator(returns, mode=args.mode, threshold=args.threshold)
                print(f"  valid trading days in range: {int(valid[code].sum())}")
                print(f"  'up' days (usable AS A LEADER): {int(up[code].sum())}")
        except Exception as exc:  # noqa: BLE001 - report and move to the next code
            print(f"  live check failed: {exc}")


def main():
    args = parse_args()
    codes = [normalize_code(c) for c in args.codes]
    print("checking codes:", codes)

    if args.pairs:
        print(f"\n--- final/merged pairs file: {args.pairs} ---")
        check_pairs_file(Path(args.pairs), codes, label="merged")

    if args.by_sector_dir:
        print(f"\n--- per-sector pairs files under {args.by_sector_dir} (before the global budget cap) ---")
        scan_by_sector_dir(args.by_sector_dir, codes)

    if args.source == "xtdata":
        live_checks(codes, args)
    else:
        print("\n(pass --source xtdata --kind {leadlag,limitup,intraday} to also check "
              "ST status, sector mapping, and raw trigger-event counts)")


if __name__ == "__main__":
    main()
