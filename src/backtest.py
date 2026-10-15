"""Rolling-origin backtest of Prophet vs SARIMA vs Holt-Winters.

Mirrors the real exercise (forecasting next calendar year from data through September):
each fold trains on data up to a past month-end and is scored on months 4-15 after it.
Stores enter a fold once they have 24 full months of history at the origin.

Writes outputs/backtest_forecasts.csv and outputs/backtest_metrics.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecasting import FORECASTERS, MODELS, OUTPUTS, bias, load_stores, mape, wape

ORIGINS = ["2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"]
LAST_COMPLETE_MONTH = pd.Period("2026-09", "M")
MIN_HISTORY_MONTHS = 24
SKIP_MONTHS = 3  # score months 4-15 after the origin
HORIZON = 15
ENSEMBLE = "Average of 3"


def eligible(sd, origin: pd.Timestamp) -> bool:
    return (origin.to_period("M") - sd.first_full_month).n + 1 >= MIN_HISTORY_MONTHS


def run_task(sd, origin, model):
    last = min(origin.to_period("M") + HORIZON, LAST_COMPLETE_MONTH)
    point, _ = FORECASTERS[model](sd, origin, last)
    return sd.store, origin, model, point


def main(n_jobs: int = 4) -> None:
    stores = load_stores()
    tasks = [
        (sd, pd.Timestamp(o), m)
        for o in ORIGINS for sd in stores.values() if eligible(sd, pd.Timestamp(o)) for m in MODELS
    ]
    print(f"{len(tasks)} fits ({len(tasks) // len(MODELS)} store-folds)")
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_task)(*t) for t in tasks)

    rows = []
    for store, origin, model, point in results:
        o = origin.to_period("M")
        for month, value in point.items():
            h = (month - o).n
            if h > SKIP_MONTHS:
                rows.append((origin.date(), store, month, h, model, value))
    fc = pd.DataFrame(rows, columns=["origin", "store", "month", "horizon", "model", "forecast"])
    fc = fc.pivot_table(index=["origin", "store", "month", "horizon"], columns="model",
                        values="forecast").reset_index()
    fc[ENSEMBLE] = fc[MODELS].mean(axis=1)
    actual = {s: sd.monthly_actual() for s, sd in stores.items()}
    fc.insert(4, "actual", [actual[s].get(m, np.nan) for s, m in zip(fc.store, fc.month)])
    fc = fc.sort_values(["origin", "store", "month"]).reset_index(drop=True)
    fc.to_csv(OUTPUTS / "backtest_forecasts.csv", index=False)

    models = MODELS + [ENSEMBLE]
    brand = fc.groupby(["origin", "month"])[["actual"] + models].sum().reset_index()
    metrics = []
    for origin, g in fc.groupby("origin"):
        b = brand[brand.origin == origin]
        for m in models:
            metrics.append({
                "origin": str(origin), "model": m, "stores": g.store.nunique(),
                "months": b.month.nunique(),
                "store_month_wape": wape(g.actual, g[m]),
                "brand_month_mape": mape(b.actual, b[m]),
                "brand_window_error": bias(b.actual, b[m]),
            })
    for m in models:
        per_fold = [x for x in metrics if x["model"] == m]
        metrics.append({
            "origin": "ALL", "model": m, "stores": fc.store.nunique(), "months": len(brand),
            "store_month_wape": wape(fc.actual, fc[m]),
            "brand_month_mape": mape(brand.actual, brand[m]),
            "brand_window_error": np.mean([x["brand_window_error"] for x in per_fold]),
            "brand_window_abs_error": np.mean([abs(x["brand_window_error"]) for x in per_fold]),
        })
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUTPUTS / "backtest_metrics.csv", index=False)

    pd.set_option("display.width", 200)
    print(metrics.round(4).to_string(index=False))
    by_store = fc.groupby("store").apply(lambda g: pd.Series({m: wape(g.actual, g[m]) for m in models}),
                                         include_groups=False)
    print("\nStore-month WAPE by store:\n" + by_store.round(3).to_string())
    by_h = fc.groupby("horizon").apply(lambda g: pd.Series({m: wape(g.actual, g[m]) for m in models}),
                                       include_groups=False)
    print("\nStore-month WAPE by horizon:\n" + by_h.round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
