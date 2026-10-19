"""Unpack the channel extracts (daily/sql/channels_by_day.sql, one MCP result file per window) into one row per
store-day with sales in store, by online pickup and by delivery, and check them against the daily sales the models
use (data/raw/parakeet_toast_daily.csv).

The result files are external data: run this with `python -I` and pass their paths as arguments.

    .venv/bin/python -I daily/channel_data.py ch_2022.txt ch_2023.txt ch_2024.txt ch_2025.txt ch_2026.txt
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw" / "parakeet_toast_daily.csv"
OUT = HERE / "data" / "toast_channels_daily.csv"
CHANNELS = ["in_store", "pickup", "delivery"]


def read_result(path: Path) -> list[dict]:
    text = path.read_text()
    return json.loads(text[:text.rindex("}") + 1])["rows"]  # the client appends a note after the JSON


def unpack(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        days = [d.split("=") for d in r["x"].split("|")]
        if len(days) != r["n"]:
            raise ValueError(f"{r['v']} {r['m']}: {len(days)} days packed, {r['n']} counted")
        for day, vals in days:
            ins, pick, dlv = (int(v) for v in vals.split("/"))
            out.append({"store": r["v"], "date": pd.Timestamp(f"{r['m']}-{int(day):02d}"), "in_store": ins,
                        "pickup": pick, "delivery": dlv, "month_total": r["tot"], "month_days": r["n"]})
    return pd.DataFrame(out)


def main(paths: list[str]) -> None:
    ch = pd.concat([unpack(read_result(Path(p))) for p in paths], ignore_index=True)
    if ch.duplicated(["store", "date"]).any():
        raise ValueError("a store-day appears in two extracts")
    ch["total"] = ch[CHANNELS].sum(axis=1)

    # whole dollars per channel: each month adds up to its exact total within half a dollar a channel-day
    m = ch.groupby(["store", ch.date.dt.to_period("M")]).agg(total=("total", "sum"), exact=("month_total", "first"),
                                                             days=("month_days", "first"))
    bad = (m.total - m.exact).abs() > 1.5 * m.days
    if bad.any():
        raise ValueError(f"month totals off:\n{m[bad]}")

    raw = pd.read_csv(RAW, parse_dates=["date"])
    both = raw.merge(ch, on=["store", "date"], how="inner")
    gap = (both.net_sales - both.total).abs()
    print(f"{len(ch):,} store-days, {ch.date.min():%Y-%m-%d} to {ch.date.max():%Y-%m-%d}")
    print(f"against the daily file: {len(both):,} store-days matched, largest gap ${gap.max():.2f} "
          f"(rounding to whole dollars per channel)")
    missing = raw[(raw.store != "Nautilus") & (raw.net_sales != 0) & (raw.date >= ch.date.min())].merge(
        ch[["store", "date"]], on=["store", "date"], how="left", indicator=True)
    missing = missing[missing._merge == "left_only"]
    print(f"store-days with sales in the daily file but no channel row: {len(missing)}")
    if gap.max() > 1.5 or len(missing):
        raise ValueError("the channel extract does not match the daily file")

    OUT.parent.mkdir(exist_ok=True)
    ch.sort_values(["store", "date"])[["store", "date", *CHANNELS]].to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"wrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main(sys.argv[1:])
