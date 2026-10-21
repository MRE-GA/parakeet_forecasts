"""Backtests for the hourly patterns of the week templates (Toast hourly data, Oct 2024 to Oct 14, 2026).

The metric is the hourly error given the true day: each actual day's sales are split into hours with a method's
pattern and compared hour by hour, sum |actual - split| / sum actual, over normal trading days. It is exactly the
error the hourly split adds to a day's figure. Checks: each actual hour's sales divided by a method's average check,
compared with the hour's actual checks.

1. Which pattern (learn Oct 2024 - Oct 14, 2025, score Oct 15, 2025 - Oct 14, 2026): one per weekday for the whole
   year, one per type of week and weekday, per type and group of days (Mon-Thu, Fri, Sat, Sun), per type, and the
   average of the first two; for special days, their own pattern from the year before against their type's.
2. How much recent days weigh in the weekday pattern, and the average check (learn Oct 2024 - Apr 14, 2026, score
   Apr 15 - Oct 14, 2026).

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
GROUP = {0: "Mon-Thu", 1: "Mon-Thu", 2: "Mon-Thu", 3: "Mon-Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def score(test: pd.DataFrame, pred_share: dict, key) -> pd.DataFrame:
    """Hourly error per store-day: the day's actual sales split with the pattern found under key(day)."""
    rows = []
    for (s, d), g in test.groupby(["store", "date"]):
        share = pred_share(s, key(g.iloc[0]))
        if share is None:
            continue
        a = g.set_index("hour").net_sales.reindex(share.index).fillna(0).to_numpy()
        rows.append((s, d, np.abs(a - share.to_numpy() * a.sum()).sum(), a.sum()))
    return pd.DataFrame(rows, columns=["store", "date", "err", "sales"])


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    weeks = classify()
    h = H.load()
    typ = {s: weeks[s].label for s in STORES}
    h["type"] = [typ[s][w] for s, w in zip(h.store, h.week)]
    h["group"] = h.dow.map(GROUP)
    results = []

    # ---------------------------------------------------------------- 1. which pattern
    cut = pd.Timestamp("2025-10-14")
    for s in STORES:
        hrs = H.store_hours(h, s, cut)
        g = H.folded(h[h.date <= pd.Timestamp("2026-10-14")], s, hrs)
        g["type"] = typ[s].reindex(g.week).to_numpy()
        g["group"] = g.dow.map(GROUP)
        tr = g[(g.date <= cut) & g.normal]
        te = g[(g.date > cut) & g.normal]
        tr_reg, te_reg = tr[tr.special == ""], te[te.special == ""]
        cache = {}

        def pat(by, k, src=tr_reg):
            if (by, k) not in cache:
                sel = src
                for col, v in zip(by, k):
                    sel = sel[sel[col] == v]
                cache[(by, k)] = (pd.Series(H.pattern(sel, hrs, cut, None), index=hrs.shown)
                                  if sel.date.nunique() >= 3 else None)
            return cache[(by, k)]

        methods = {
            "weekday": lambda r: pat(("dow",), (r.dow,)),
            "type x weekday": lambda r: pat(("type", "dow"), (r.type, r.dow)),
            "type x day group": lambda r: pat(("type", "group"), (r.type, r.group)),
            "type": lambda r: pat(("type",), (r.type,)),
        }
        for name, f in methods.items():
            z = score(te_reg, lambda _s, k: k, f)
            results.append({"test": "1 pattern", "store": s, "method": name, "err": z.err.sum(),
                            "sales": z.sales.sum()})
        def blend(r):
            a, b = methods["weekday"](r), methods["type x weekday"](r)
            return None if a is None or b is None else (a + b) / 2
        z = score(te_reg, lambda _s, k: k, blend)
        results.append({"test": "1 pattern", "store": s, "method": "average of weekday and type x weekday",
                        "err": z.err.sum(), "sales": z.sales.sum()})
        # special days: their own pattern (same special day the year before) against their type's
        te_sp = te[te.special != ""]
        tr_sp = tr[tr.special != ""]
        for name, f in {"special day: type x weekday": methods["type x weekday"],
                        "special day: its own pattern": lambda r: (pat(("special",), (r.special,), tr_sp)
                                                                   if r.special in set(tr_sp.special) else
                                                                   methods["type x weekday"](r))}.items():
            z = score(te_sp, lambda _s, k: k, f)
            results.append({"test": "1 special days", "store": s, "method": name, "err": z.err.sum(),
                            "sales": z.sales.sum()})

    # ---------------------------------------------------------------- 2. recency and the average check
    cut = pd.Timestamp("2026-04-14")
    for s in STORES:
        hrs = H.store_hours(h, s, cut)
        g = H.folded(h, s, hrs)
        g["type"] = typ[s].reindex(g.week).to_numpy()
        tr = g[(g.date <= cut) & g.normal & (g.special == "")]
        te = g[(g.date > cut) & g.normal & (g.special == "")]
        for hl in [None, 730, 365, 180, 90, 45]:
            cache = {}

            def f(r, hl=hl, cache=cache):  # the pattern test 1 chose: one per weekday
                k = r.dow
                if k not in cache:
                    sel = tr[tr.dow == r.dow]
                    cache[k] = pd.Series(H.pattern(sel, hrs, cut, hl), index=hrs.shown) if len(sel) else None
                return cache[k]
            z = score(te, lambda _s, k: k, f)
            results.append({"test": "2 recency", "store": s, "method": f"half-life {hl or 'none'}",
                            "err": z.err.sum(), "sales": z.sales.sum()})
        # average check by hour and weekday over the last n months, or by type too
        for label, months, by_type in [("check: last 3 months", 3, False), ("check: last 6 months", 6, False),
                                       ("check: last 12 months", 12, False), ("check: all 18 months", 19, False),
                                       ("check: by type, all 18 months", 19, True)]:
            src = tr[tr.date > cut - pd.DateOffset(months=months)]
            err = tot = bias = 0.0
            for keys, t in te.groupby(["type", "dow"] if by_type else ["dow"]):
                sel = src
                if by_type:
                    sel = sel[(sel.type == keys[0]) & (sel.dow == keys[1])]
                    if sel.empty:
                        sel = src[src.dow == keys[1]]
                else:
                    sel = sel[sel.dow == keys[0]]
                ac = pd.Series(H.avg_check(sel, hrs), index=hrs.shown)
                pred = t.net_sales / t.hour.map(ac)
                err += (t.checks - pred).abs().sum()
                bias += (pred - t.checks).sum()
                tot += t.checks.sum()
            results.append({"test": "2 average check", "store": s, "method": label, "err": err, "sales": tot,
                            "bias": bias})

    res = pd.DataFrame(results)
    res["store"] = res.store.map(DISPLAY)
    res.to_csv(OUTDIR / "backtest_hourly.csv", index=False)
    tot = res.groupby(["test", "method"], sort=False)[["err", "sales", "bias"]].sum()
    tot["error"] = (tot.err / tot.sales * 100).round(2)
    tot["bias"] = (tot.bias / tot.sales * 100).round(2)
    print(tot[["error", "bias"]].to_string())
    by = res.assign(e=res.err / res.sales * 100).pivot_table(index=["test", "method"], columns="store", values="e",
                                                             sort=False).round(1)
    print(by.to_string())


if __name__ == "__main__":
    main()
