"""Build the Parakeet Cafe daily sales dataset from MCP Andersons query dumps.

The extraction query (data/sql/extract_parakeet_daily.sql) returns one row per
store x month with the daily values packed as "day=net/checks|day=net/checks|...",
plus monthly checksums (n days, total net, total checks). This script unpacks
the rows, validates every month against its checksums and writes a tidy CSV.

Usage:
    python src/build_dataset.py dump1.txt [dump2.txt ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "parakeet_toast_daily.csv"
FOOTER = "…[truncado por límite del perfil]"


def load_dump(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.endswith(FOOTER):
        text = text[: -len(FOOTER)].strip()
    payload = json.loads(text)  # raises if the server cut the JSON short
    if payload.get("truncated"):
        raise ValueError(f"{path.name}: row limit hit, re-run with a smaller window")
    return payload["rows"]


def unpack(row: dict) -> list[dict]:
    year, month = map(int, row["m"].split("-"))
    records = []
    for item in row["x"].split("|"):
        day, values = item.split("=")
        net, checks = values.split("/")
        records.append(
            {
                "store": row["v"],
                "date": pd.Timestamp(year, month, int(day)),
                "net_sales": float(net),
                "checks": int(checks),
            }
        )
    # checksums computed in SQL
    assert len(records) == row["n"], (row["v"], row["m"], "day count")
    assert abs(sum(r["net_sales"] for r in records) - row["tot"]) < 0.01 * row["n"] + 0.01, (
        row["v"], row["m"], "net total")
    assert sum(r["checks"] for r in records) == row["tchk"], (row["v"], row["m"], "checks total")
    return records


def main(paths: list[str]) -> None:
    records = []
    n_rows = 0
    for p in paths:
        rows = load_dump(Path(p))
        n_rows += len(rows)
        for row in rows:
            records.extend(unpack(row))
    df = pd.DataFrame(records).sort_values(["store", "date"]).reset_index(drop=True)
    dupes = df.duplicated(["store", "date"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicated store-days")
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"{n_rows} store-months -> {len(df):,} store-days -> {OUT.relative_to(ROOT)}")
    print(df.groupby("store").agg(first=("date", "min"), last=("date", "max"), days=("date", "size"),
                                  net_sales=("net_sales", "sum")).round(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
