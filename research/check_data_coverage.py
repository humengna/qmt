"""Report what bar history is ALREADY CACHED locally, per sector, without downloading.

Answers the question you need before any `--no-download` run: which sectors can I
actually mine right now, and over what date range? xtdata caches each (symbol, period)
separately and `get_market_data_ex` simply returns less - or nothing - for what was
never fetched, rather than complaining, so an uncached sector otherwise only shows up
as a confusing empty result much later in the pipeline.

Reads through the very same helper the mining uses
(`intraday.data.fetch_intraday_close_panels_xtdata` with download=False), so what this
reports is exactly what a mining run would see.

Usage::

    # every default SW1 sector, 5-minute bars, whatever range you plan to mine
    python research/check_data_coverage.py --period 5m --start 20230101 --end 20260922

    # just a few sectors, and check every symbol instead of a sample
    python research/check_data_coverage.py --period 5m --sectors SW1电子 SW1银行 --sample 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from intraday import data as idd
from leadlag import data as ld


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sectors", nargs="*", default=None,
                    help="sector/board names to check (default: leadlag.data.DEFAULT_SW_L1_SECTORS). "
                         "If a name reports 0 stocks it does not exist in THIS client - dump the "
                         "real ones with leadlag.data.list_sectors_xtdata().")
    p.add_argument("--period", default="5m")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--sample", type=int, default=5,
                    help="how many symbols per sector to actually read (0 = all). Sampling keeps "
                         "this quick; a sector's symbols are normally downloaded together, so a "
                         "handful is a good proxy for the whole sector.")
    p.add_argument("--output", default="research/output/data_coverage.csv")
    return p.parse_args()


def summarize_coverage(minute_close: pd.DataFrame) -> dict:
    """Turn a fetched panel into one coverage row. Pure - no xtdata involved."""
    if minute_close is None or minute_close.empty or minute_close.shape[1] == 0:
        return {"with_data": 0, "first_bar": None, "last_bar": None, "median_bars": 0}

    per_symbol_bars = minute_close.notna().sum()
    with_data = int((per_symbol_bars > 0).sum())
    if with_data == 0:
        return {"with_data": 0, "first_bar": None, "last_bar": None, "median_bars": 0}

    covered = minute_close.loc[:, per_symbol_bars > 0]
    non_empty_rows = covered.dropna(how="all").index
    return {
        "with_data": with_data,
        "first_bar": str(non_empty_rows.min()) if len(non_empty_rows) else None,
        "last_bar": str(non_empty_rows.max()) if len(non_empty_rows) else None,
        "median_bars": int(per_symbol_bars[per_symbol_bars > 0].median()),
    }


def check_sector(sector: str, args) -> dict:
    row = {"sector": sector, "stocks_in_sector": 0, "sampled": 0,
           "with_data": 0, "first_bar": None, "last_bar": None, "median_bars": 0, "note": ""}
    try:
        stock_list = ld.get_full_market_stock_list_xtdata((sector,))
    except Exception as exc:  # noqa: BLE001 - keep checking the other sectors
        row["note"] = f"sector lookup failed: {exc}"
        return row

    row["stocks_in_sector"] = len(stock_list)
    if not stock_list:
        row["note"] = "board name not found in this client"
        return row

    sampled = stock_list if args.sample <= 0 else stock_list[: args.sample]
    row["sampled"] = len(sampled)
    try:
        minute_close, _ = idd.fetch_intraday_close_panels_xtdata(
            sampled, start_time=args.start, end_time=args.end, period=args.period, download=False,
        )
    except Exception as exc:  # noqa: BLE001 - an empty cache can surface as a parse error
        row["note"] = f"read failed (usually means nothing cached): {exc}"
        return row

    row.update(summarize_coverage(minute_close))
    if row["with_data"] == 0:
        row["note"] = f"no {args.period} bars cached for this range"
    return row


def main():
    args = parse_args()
    sectors = args.sectors or ld.DEFAULT_SW_L1_SECTORS
    print(f"checking {len(sectors)} sector(s) for cached {args.period} bars over "
          f"{args.start or '(open)'}..{args.end or '(open)'}; "
          f"{'all symbols' if args.sample <= 0 else f'{args.sample} symbols sampled each'}\n")

    rows = [check_sector(s, args) for s in sectors]
    for r in rows:
        status = "OK " if r["with_data"] else "-- "
        print(f"{status}{r['sector']:<16} {r['with_data']}/{r['sampled']} symbols  "
              f"{r['first_bar'] or '-':<20} .. {r['last_bar'] or '-':<20} "
              f"median {r['median_bars']} bars  {r['note']}")

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    usable = df[df["with_data"] > 0]["sector"].tolist()
    print(f"\n{len(usable)}/{len(sectors)} sectors have cached {args.period} data")
    if usable:
        print("runnable with --no-download:")
        print(f"    --sectors {' '.join(usable)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
