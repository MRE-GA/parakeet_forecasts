"""Live check of the week templates on days after the data they were built from (Toast through Oct 14, 2026): the
days of Oct 15-19, 2026 (data/sql/extract_parakeet_hourly_flat.sql), against today's run-rate for those days and the
stores' hourly patterns (weektypes/hourly.py), the way the templates are built.

For every store and day: the actual day against the run-rate; the hours against the run-rate day split along the
pattern (sales), and the same split divided by the average check (checks); and the hours against the actual day split
along the pattern (the shape alone). Errors are sum |actual - expected| / sum actual.

    python -I weektypes/live_check.py dump.txt   -> data/raw/parakeet_toast_hourly_live.csv,
                                                    outputs/weektypes/live_check.csv
    python weektypes/live_check.py               (re-score from the saved CSV)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hourly as H  # noqa: E402
from model import DISPLAY, OUT, ROOT, RUNRATE_26, STORES  # noqa: E402
from templates import store_patterns  # noqa: E402

LIVE = ROOT / "data" / "raw" / "parakeet_toast_hourly_live.csv"
OUTDIR = OUT / "weektypes"
FOOTER = "…[truncado por límite del perfil]"


def load_dump(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    if text.endswith(FOOTER):
        text = text[: -len(FOOTER)].strip()
    payload = json.loads(text)
    if payload.get("truncated"):
        raise ValueError(f"{path.name}: row limit hit")
    return pd.DataFrame(payload["rows"])


def main(paths: list[str]) -> None:
    if paths:
        live = pd.concat([load_dump(Path(p)) for p in paths]).sort_values(["store", "date", "hour"])
        assert not live.duplicated(["store", "date", "hour"]).any()
        live.to_csv(LIVE, index=False)
    live = pd.read_csv(LIVE, parse_dates=["date"])
    back = {v: k for k, v in DISPLAY.items()}
    rr = pd.read_csv(RUNRATE_26, parse_dates=["date"])
    rr = rr.assign(store=rr.store.map(back)).set_index(["store", "date"]).runrate
    h = H.load()
    rows = []
    for s in STORES:
        hours, pat, chk, _ = store_patterns(h, s)
        g = live[live.store == s].copy()
        g["hour"] = hours.fold(g.hour.to_numpy())
        for d, gd in g.groupby("date"):
            a = gd.groupby("hour")[["net_sales", "checks"]].sum().reindex(hours.shown).fillna(0)
            p, c = pat[d.dayofweek], chk[d.dayofweek]
            run = rr[(s, d)]
            rows.append({"store": DISPLAY[s], "date": d.date(), "actual": a.net_sales.sum(), "runrate": run,
                         "hour_err": np.abs(a.net_sales - p * run).sum(),
                         "shape_err": np.abs(a.net_sales - p * a.net_sales.sum()).sum(),
                         "checks": a.checks.sum(), "check_err": np.abs(a.checks - p * run / c).sum(),
                         "checks_expected": (p * run / c).sum()})
    res = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTDIR / "live_check.csv", index=False)
    by = res.groupby("store", sort=False)[["actual", "runrate", "hour_err", "shape_err", "checks", "check_err",
                                            "checks_expected"]].sum()
    by.loc["All ten"] = by.sum()
    out = pd.DataFrame({"sales vs run-rate": by.actual / by.runrate - 1, "hourly error": by.hour_err / by.actual,
                        "hourly shape error": by.shape_err / by.actual,
                        "checks vs expected": by.checks / by.checks_expected - 1,
                        "hourly check error": by.check_err / by.checks})
    print(f"{res.date.min()} to {res.date.max()}, {len(res)} store-days")
    print(out.mul(100).round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
