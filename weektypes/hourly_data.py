"""Unpack the hourly Toast extracts (data/sql/extract_parakeet_hourly.sql, one MCP result file per store) into
data/raw/parakeet_toast_hourly.csv.gz: net sales (whole dollars) and checks by store, day and hour opened.

Every month is checked against the checksums computed in SQL, and every day against the daily file
(data/raw/parakeet_toast_daily.csv, same net-sales definition, in cents): the hours of a day add up to it within the
rounding to whole dollars.

    python -I weektypes/hourly_data.py dump_store1.txt [dump_store2.txt ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "parakeet_toast_hourly.csv.gz"
DAILY = ROOT / "data" / "raw" / "parakeet_toast_daily.csv"
FOOTER = "…[truncado por límite del perfil]"


def load_dump(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.endswith(FOOTER):
        text = text[: -len(FOOTER)].strip()
    payload = json.loads(text)  # raises if the client cut the JSON short
    if payload.get("truncated"):
        raise ValueError(f"{path.name}: row limit hit")
    return payload["rows"]


def unpack(row: dict) -> list[dict]:
    year, month = map(int, row["m"].split("-"))
    records = []
    for day_part in row["x"].split("|"):
        day, cells = day_part.split("@")
        date = pd.Timestamp(year, month, int(day))
        for cell in cells.split(","):
            hour, values = cell.split("=")
            net, checks = values.split("/")
            records.append({"store": row["v"], "date": date, "hour": int(hour), "net_sales": int(net),
                            "checks": int(checks)})
    assert len(records) == row["n"], (row["v"], row["m"], "cell count")
    assert sum(r["net_sales"] for r in records) == row["tot"], (row["v"], row["m"], "net total")
    assert sum(r["checks"] for r in records) == row["tchk"], (row["v"], row["m"], "checks total")
    return records


def main(paths: list[str]) -> None:
    rows = [r for p in paths for r in load_dump(Path(p))]
    h = pd.DataFrame([rec for r in rows for rec in unpack(r)]).sort_values(["store", "date", "hour"])
    assert not h.duplicated(["store", "date", "hour"]).any()
    # against the daily file: each day's hours add up to its net sales (rounding: at most $0.50 an hour)
    day = h.groupby(["store", "date"]).agg(net=("net_sales", "sum"), chk=("checks", "sum"), n=("hour", "size"))
    d = pd.read_csv(DAILY, parse_dates=["date"]).set_index(["store", "date"])
    both = day.join(d, how="inner")
    gap = (both.net - both.net_sales).abs()
    assert (gap <= 0.5 * both.n + 0.01).all(), both[gap > 0.5 * both.n + 0.01].head()
    assert (both.chk == both.checks).all()
    stores = d.index.get_level_values("store").isin(h.store.unique())  # Nautilus is not extracted
    missing = d.loc[stores & (d.index.get_level_values("date") >= h.date.min())].index.difference(day.index)
    print(f"{len(h):,} store-hours, {len(day):,} store-days from {h.date.min():%Y-%m-%d} to {h.date.max():%Y-%m-%d}; "
          f"days in both files {len(both):,} (largest gap ${gap.max():.2f}, checks equal); "
          f"days in the daily file only: {len(missing)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    h.assign(date=h.date.dt.strftime("%Y-%m-%d")).to_csv(OUT, index=False)
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
