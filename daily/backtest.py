"""Backtest: which method splits a month into its days best?

Six forecast origins (Dec 2024 to Mar 2026, quarterly); each method learns only from data up to the origin and
splits each of the next 12 months (to Sep 2026) into days. Metric: daily WAPE given the true month (common.py).

    python daily/backtest.py   -> daily/outputs/backtest_store_month.csv (error by origin, store, month, method)
                                  daily/outputs/backtest_days.pkl (day level, for breakdowns; not committed)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LAST_DAY, LGBM, OUT, STORES, Structural, build_panel, m_prophet, shape_errors,  # noqa: E402
                    wape)
from forecasting import load_stores  # noqa: E402

ORIGINS = [pd.Timestamp(d) for d in ("2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31",
                                     "2026-03-31")]
LAST_MONTH = pd.Period("2026-09", "M")
PROPHET_VARIANTS = {"prophet_tuned": {}, "prophet_free": {"free_weekly": True},
                    "prophet_summer": {"summer_weekly": True}}


def prophet_job(sd, origin, dates, variant):
    return m_prophet(sd, origin, dates, **PROPHET_VARIANTS[variant])


def main() -> None:
    t0 = time.time()
    panel = build_panel()
    stores = load_stores(LAST_DAY)
    parts = []
    for origin in ORIGINS:
        months = pd.period_range(origin.to_period("M") + 1, min(origin.to_period("M") + 12, LAST_MONTH), freq="M")
        test = panel[panel.month.isin(months)].copy()
        test["origin"] = origin
        test["horizon"] = (test.month - origin.to_period("M")).apply(lambda x: x.n)
        # Prophet, per store and variant
        jobs = [(s, v) for s in STORES for v in PROPHET_VARIANTS if (test.store == s).any()]
        res = Parallel(n_jobs=4)(delayed(prophet_job)(stores[s], origin, pd.DatetimeIndex(test[test.store == s].date), v)
                                 for s, v in jobs)
        for (s, v), w in zip(jobs, res):
            test.loc[test.store == s, v] = w
        # empirical methods
        test["flat"] = test.open.to_numpy()
        test["dow52"] = Structural(panel, origin, seasonal=False, specials=False, use_curve=False).weights(test)
        st = Structural(panel, origin)
        test["structural"] = st.weights(test)
        test["structural_nocurve"] = Structural(panel, origin, use_curve=False).weights(test)
        test["lgbm"] = LGBM(panel, origin).weights(test)
        test["ensemble"] = np.sqrt(test.structural * test.lgbm)
        parts.append(test)
        print(f"origin {origin.date()}: {len(months)} months, {time.time() - t0:.0f}s", flush=True)
    days = pd.concat(parts, ignore_index=True)
    methods = ["flat", "dow52", "prophet_tuned", "prophet_free", "prophet_summer", "structural_nocurve", "structural",
               "lgbm", "ensemble"]
    rows = []
    for m in methods:
        zz = []
        for o, g in days.groupby("origin"):
            e = shape_errors(g, m)
            e["origin"] = o
            zz.append(e)
        z = pd.concat(zz)
        days.loc[z.index, f"err_{m}"] = z.err
        g = z.groupby(["origin", "store", "month", "horizon"]).agg(err=("err", "sum"), sales=("sales", "sum"))
        g["method"] = m
        rows.append(g.reset_index())
        print(f"{m:20s} daily WAPE {wape(z) * 100:5.2f}%")
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "backtest_store_month.csv", index=False, float_format="%.2f")
    days.to_pickle(OUT / "backtest_days.pkl")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
