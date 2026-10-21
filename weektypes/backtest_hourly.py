"""Backtests for the hourly patterns of the week templates (Toast hourly data, Oct 2024 to Oct 14, 2026).

Two errors, over normal trading days. Day: each actual day's sales split into hours with a method's pattern and
compared hour by hour, sum |actual - split| / sum actual: the error the split adds to a single day, mostly the noise
of one hour on one day. Template: the same, summed first over the days of each type of week and day of the week
(store x type x weekday x hour), which is what a template shows: whether the average week of a type has its sales in
the right hours (say, more into the afternoon in summer). Checks: each actual hour's sales divided by a method's
average check, against the hour's actual checks. Past days take the type of their week of the 52 (model.window_week).

1. Which pattern (learn Oct 2024 - Oct 14, 2025, score Oct 15, 2025 - Oct 14, 2026): one per weekday for the whole
   year; one per type of week and weekday; the weekday's pattern times its type's shift by hour (learned on all the
   type's days), in full or half, or a shift for weekdays and weekends apart; for special days, their own pattern
   from the year before against their type's.
2. How much recent days weigh, with and without the shift, and the average check (learn Oct 2024 - Apr 14, 2026,
   score Apr 15 - Oct 14, 2026: spring, summer and early fall).

    python weektypes/backtest_hourly.py   -> outputs/weektypes/backtest_hourly.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hourly as H  # noqa: E402
from model import DISPLAY, OUT, STORES, classify  # noqa: E402

OUTDIR = OUT / "weektypes"


def score(test: pd.DataFrame, hours: H.Hours, share_of) -> dict:
    """Day and template errors: every test day split with share_of(first row of the day) (None: skip the day)."""
    act, pred, keys = [], [], []
    for (_, _), g in test.groupby(["store", "date"]):
        r = g.iloc[0]
        p = share_of(r)
        if p is None:
            continue
        a = g.set_index("hour").net_sales.reindex(hours.shown).fillna(0).to_numpy()
        act.append(a)
        pred.append(np.asarray(p) * a.sum())
        keys.append((r.type, r.dow))
    act, pred = np.array(act), np.array(pred)
    k = pd.MultiIndex.from_tuples(keys)
    agg_a = pd.DataFrame(act, index=k).groupby(level=[0, 1]).sum()
    agg_p = pd.DataFrame(pred, index=k).groupby(level=[0, 1]).sum()
    # parts of the day, by type of week (all days of the week together): morning, midday, afternoon
    parts = [[i for i, hr in enumerate(hours.shown) if lo <= hr <= hi] for lo, hi in H.DAYPARTS]
    typ = [t for t, _ in keys]
    dp_a = pd.DataFrame({j: act[:, p].sum(1) for j, p in enumerate(parts)}).groupby(typ).sum()
    dp_p = pd.DataFrame({j: pred[:, p].sum(1) for j, p in enumerate(parts)}).groupby(typ).sum()
    return {"err": np.abs(act - pred).sum(), "agg_err": (agg_a - agg_p).abs().to_numpy().sum(),
            "dp_err": (dp_a - dp_p).abs().to_numpy().sum(), "sales": act.sum()}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    weeks = classify()
    h = H.load()
    results = []

    def add(test, s, method, z):
        results.append({"test": test, "store": s, "method": method, **z})

    # ---------------------------------------------------------------- 1. which pattern
    cut = pd.Timestamp("2025-10-14")
    for s in STORES:
        hrs = H.store_hours(h, s, cut)
        g = H.folded(h[h.date <= pd.Timestamp("2026-10-14")], s, hrs)
        g["type"] = weeks[s].label.reindex(g.week).to_numpy()
        g["wkend"] = g.dow >= 5
        tr = g[(g.date <= cut) & g.normal]
        te = g[(g.date > cut) & g.normal]
        tr_reg, te_reg = tr[tr.special == ""], te[te.special == ""]
        flat = H.fit_patterns(tr_reg, hrs, cut, None, strength=0)
        sep = {k: H.pattern(x, hrs, cut, None) for k, x in tr_reg.groupby(["type", "dow"]) if x.date.nunique() >= 3}
        fits = {"weekday x type shift by hour": H.fit_patterns(tr_reg, hrs, cut, None),
                "weekday x half the type shift by hour": H.fit_patterns(tr_reg, hrs, cut, None, strength=0.5),
                "weekday x type shift by part of the day": H.fit_patterns(tr_reg, hrs, cut, None, dayparts=True),
                "weekday x half the type shift by part of the day":
                    H.fit_patterns(tr_reg, hrs, cut, None, strength=0.5, dayparts=True),
                "weekday x type shift by hour, weekdays and weekends apart":
                    H.fit_patterns(tr_reg, hrs, cut, None, by=("type", "wkend"))}
        add("1 pattern", s, "weekday", score(te_reg, hrs, lambda r: flat.get(r.dow)))
        add("1 pattern", s, "type x weekday", score(te_reg, hrs, lambda r: sep.get((r.type, r.dow))))
        for name, f in fits.items():
            add("1 pattern", s, name, score(te_reg, hrs, lambda r, f=f: f.get(r.dow, tuple(r[c] for c in f.by))))
        # special days: their own pattern (same special day the year before) against their type's
        te_sp, tr_sp = te[te.special != ""], tr[tr.special != ""]
        own = {k: H.pattern(x, hrs, cut, None) for k, x in tr_sp.groupby("special")}
        best = fits["weekday x type shift by hour"]
        add("1 special days", s, "special day: the weekday's pattern with its type's shift",
            score(te_sp, hrs, lambda r: best.get(r.dow, (r.type,))))
        add("1 special days", s, "special day: its own pattern",
            score(te_sp, hrs, lambda r: own.get(r.special, best.get(r.dow, (r.type,)))))

    # ---------------------------------------------------------------- 2. recency, the shift again, the average check
    cut = pd.Timestamp("2026-04-14")
    for s in STORES:
        hrs = H.store_hours(h, s, cut)
        g = H.folded(h, s, hrs)
        g["type"] = weeks[s].label.reindex(g.week).to_numpy()
        tr = g[(g.date <= cut) & g.normal & (g.special == "")]
        te = g[(g.date > cut) & g.normal & (g.special == "")]
        for hl in [None, 365, 180, 90]:
            for strength, dp, label in [(0, False, "weekday"), (1, False, "weekday x type shift by hour"),
                                        (0.5, False, "weekday x half the type shift by hour"),
                                        (1, True, "weekday x type shift by part of the day"),
                                        (0.5, True, "weekday x half the type shift by part of the day")]:
                f = H.fit_patterns(tr, hrs, cut, hl, strength=strength, dayparts=dp)
                add("2 recency", s, f"{label}, half-life {hl or 'none'}",
                    score(te, hrs, lambda r, f=f: f.get(r.dow, (r.type,))))
        # average check by hour and weekday over the last n months, or by type too
        for label, months, by_type in [("check: last 3 months", 3, False), ("check: last 6 months", 6, False),
                                       ("check: last 12 months", 12, False), ("check: all 18 months", 19, False),
                                       ("check: by type, all 18 months", 19, True)]:
            src = tr[tr.date > cut - pd.DateOffset(months=months)]
            err = tot = bias = 0.0
            for keys, t in te.groupby(["type", "dow"] if by_type else ["dow"]):
                if by_type:
                    sel = src[(src.type == keys[0]) & (src.dow == keys[1])]
                    sel = sel if not sel.empty else src[src.dow == keys[1]]
                else:
                    sel = src[src.dow == keys[0]]
                ac = pd.Series(H.avg_check(sel, hrs), index=hrs.shown)
                pred = t.net_sales / t.hour.map(ac)
                err += (t.checks - pred).abs().sum()
                bias += (pred - t.checks).sum()
                tot += t.checks.sum()
            add("2 average check", s, label, {"err": err, "agg_err": np.nan, "dp_err": np.nan, "sales": tot,
                                              "bias": bias})

    res = pd.DataFrame(results)
    res["store"] = res.store.map(DISPLAY)
    res.to_csv(OUTDIR / "backtest_hourly.csv", index=False)
    tot = res.groupby(["test", "method"], sort=False)[["err", "agg_err", "dp_err", "sales", "bias"]].sum(min_count=1)
    out = pd.DataFrame({"day error": tot.err / tot.sales, "template error": tot.agg_err / tot.sales,
                        "part-of-day error": tot.dp_err / tot.sales, "bias": tot.bias / tot.sales}).mul(100).round(2)
    print(out.to_string())
    by = res.assign(e=res.dp_err / res.sales * 100).pivot_table(index=["test", "method"], columns="store", values="e",
                                                                sort=False).round(2)
    print("part-of-day error by store:")
    print(by.to_string())


if __name__ == "__main__":
    main()
