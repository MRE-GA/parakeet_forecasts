"""The week templates: for every store, each type's average week and each holiday week's own week, by day and hour, in
sales (whole dollars) and checks.

A type's day is the average of its weeks' run-rate on that day of the week (holiday weeks left out); a holiday week's
days are its own run-rate days. Days are in whole dollars and split into hours in whole dollars that add up to the day
(largest remainder), along the store's weekday pattern, or a special day's own pattern on its day
(weektypes/hourly.py). Checks are each hour's sales divided by the store's average check for that hour and day of the
week (the last 12 months; a special day's own where it has one), rounded to whole checks.

    python weektypes/templates.py   -> outputs/weektypes/templates_long.csv, calendar.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hourly as H  # noqa: E402
from model import (DAYS, DISPLAY, LAST_DAY, N_WEEKS, OUT, STORES, StoreWeeks, classify, runrate_days,  # noqa: E402
                   week_start)

OUTDIR = OUT / "weektypes"


@dataclass
class Template:
    store: str
    name: str  # the type, or the holiday week's name
    kind: str  # "type" or "holiday"
    weeks: list[int]  # the weeks it is used in
    hours: list[int]
    sales: np.ndarray  # hours x 7, whole dollars
    checks: np.ndarray  # hours x 7, whole checks
    dates: list[pd.Timestamp] | None = None  # holiday weeks: the days
    special: list[str] | None = None  # holiday weeks: the special day on each day ("" if none)
    base_type: str | None = None  # holiday weeks: the type around it

    @property
    def day_sales(self) -> np.ndarray:
        return self.sales.sum(0)

    @property
    def week_sales(self) -> int:
        return int(self.sales.sum())

    @property
    def day_checks(self) -> np.ndarray:
        return self.checks.sum(0)

    @property
    def columns(self) -> list[str]:
        if self.dates is None:
            return DAYS
        return [f"{d:%a %b} {d.day}" for d in self.dates]


def split_whole(total: int, weights: np.ndarray) -> np.ndarray:
    """Whole units in proportion to the weights, adding up exactly to the total (largest remainder)."""
    raw = weights / weights.sum() * total
    base = np.floor(raw).astype(np.int64)
    base[np.argsort(-(raw - base), kind="stable")[: total - int(base.sum())]] += 1
    return base


def store_patterns(h: pd.DataFrame, store: str):
    """Hours shown, the weekday patterns and average checks (7 x hours), and each special day's own pattern and
    average check."""
    hours = H.store_hours(h, store)
    g = H.folded(h, store, hours)
    normal = g[g.normal]
    reg = normal[normal.special == ""]
    recent = reg[reg.date > LAST_DAY - pd.DateOffset(months=H.CHECK_MONTHS)]
    pat = np.array([H.pattern(reg[reg.dow == d], hours, LAST_DAY, H.RECENCY_HALF_LIFE) for d in range(7)])
    chk = np.array([H.avg_check(recent[recent.dow == d], hours) for d in range(7)])
    special = {}
    for lab, sg in normal[normal.special != ""].groupby("special"):
        fallback = reg[reg.dow == sg.dow.mode().iloc[0]]
        special[lab] = (H.pattern(sg, hours, LAST_DAY, H.RECENCY_HALF_LIFE, fallback=fallback),
                        H.avg_check(sg, hours))
    return hours, pat, chk, special


def day_cells(total: float, pattern: np.ndarray, check: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sales = split_whole(int(round(total)), pattern)
    return sales, np.rint(sales / check).astype(np.int64)


def build(weeks: dict[str, StoreWeeks] | None = None, rr: pd.DataFrame | None = None):
    rr = runrate_days() if rr is None else rr
    weeks = classify(rr) if weeks is None else weeks
    h = H.load()
    out: dict[str, list[Template]] = {}
    for s in STORES:
        sw = weeks[s]
        hours, pat, chk, special = store_patterns(h, s)
        day = rr[s]
        temps = []
        for t in sw.types:
            reg = sw.regular(t)
            avg = np.array([np.mean([day[week_start(w) + pd.Timedelta(days=d)] for w in reg]) for d in range(7)])
            cells = [day_cells(avg[d], pat[d], chk[d]) for d in range(7)]
            temps.append(Template(s, t, "type", [w for w in sw.label.index if sw.label[w] == t], hours.shown,
                                  np.column_stack([c[0] for c in cells]), np.column_stack([c[1] for c in cells])))
        hol_days = {w: dict(v) for w, v in sw.holiday_days.items()}
        for w, name in sw.holiday.items():
            dates = [week_start(w) + pd.Timedelta(days=d) for d in range(7)]
            labs = [hol_days.get(w, {}).get(d, "") for d in dates]
            cells = []
            for d, (dt, lab) in enumerate(zip(dates, labs)):
                key = lab.replace(" (observed)", "") if lab else ""
                p, c = special.get(key, (pat[d], chk[d])) if key else (pat[d], chk[d])
                cells.append(day_cells(day[dt], p, c))
            temps.append(Template(s, name, "holiday", [w], hours.shown, np.column_stack([c[0] for c in cells]),
                                  np.column_stack([c[1] for c in cells]), dates, labs, sw.label[w]))
        out[s] = temps
    return weeks, out


def calendar(weeks: dict[str, StoreWeeks]) -> pd.DataFrame:
    rows = []
    for w in range(1, N_WEEKS + 1):
        r = {"week": w, "monday": week_start(w).date(), "sunday": (week_start(w) + pd.Timedelta(days=6)).date()}
        for s in STORES:
            sw = weeks[s]
            r[DISPLAY[s]] = sw.holiday.get(w, sw.label[w])
        rows.append(r)
    return pd.DataFrame(rows)


def check_totals(weeks, temps, rr) -> pd.DataFrame:
    """Each week's template applied to the calendar, against the run-rate of the 52 weeks."""
    rows = []
    for s in STORES:
        by = {t.name: t for t in temps[s]}
        total = sum(by[weeks[s].holiday.get(w, weeks[s].label[w])].week_sales for w in range(1, N_WEEKS + 1))
        run = rr[s].sum()
        rows.append({"store": DISPLAY[s], "templates": total, "runrate": round(run, 2), "difference": total - run})
    return pd.DataFrame(rows)


def long_table(temps) -> pd.DataFrame:
    rows = []
    for s, ts in temps.items():
        for t in ts:
            for d in range(7):
                for i, hr in enumerate(t.hours):
                    rows.append({"store": DISPLAY[s], "template": t.name, "kind": t.kind,
                                 "weeks": " ".join(map(str, t.weeks)), "weekday": DAYS[d],
                                 "date": t.dates[d].date() if t.dates else "", "hour": hr,
                                 "sales": int(t.sales[i, d]), "checks": int(t.checks[i, d])})
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rr = runrate_days()
    weeks, temps = build(rr=rr)
    long_table(temps).to_csv(OUTDIR / "templates_long.csv", index=False)
    calendar(weeks).to_csv(OUTDIR / "calendar.csv", index=False)
    chk = check_totals(weeks, temps, rr)
    print(chk.to_string(index=False))
    for s in STORES:
        for t in temps[s]:
            print(f"{DISPLAY[s]:18s} {t.name:24s} weeks {len(t.weeks):2d}  ${t.week_sales:>8,}  "
                  f"checks {t.checks.sum():>6,}  hours {t.hours[0]}-{t.hours[-1]}")


if __name__ == "__main__":
    main()
