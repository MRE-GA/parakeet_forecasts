"""Check Prophet's configuration on the same backtest folds (writes outputs/prophet_sensitivity.csv)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest import HORIZON, LAST_COMPLETE_MONTH, ORIGINS, SKIP_MONTHS, eligible
from forecasting import OUTPUTS, bias, load_stores, mape, prophet_forecast, wape

VARIANTS = {
    "Multiplicative, default trend (baseline)": {},
    "Additive seasonality": {"seasonality_mode": "additive"},
    "Multiplicative, stiffer trend (cps 0.01)": {"changepoint_prior_scale": 0.01},
    "Multiplicative, flexible trend (cps 0.25)": {"changepoint_prior_scale": 0.25},
    "Log sales, additive (exponential growth)": {"seasonality_mode": "additive", "log_y": True},
}


def run(sd, origin, name, kwargs):
    last = min(origin.to_period("M") + HORIZON, LAST_COMPLETE_MONTH)
    point, _ = prophet_forecast(sd, origin, last, **kwargs)
    return name, origin, sd.store, point


def main() -> None:
    stores = load_stores()
    tasks = [(sd, pd.Timestamp(o), n, kw) for n, kw in VARIANTS.items() for o in ORIGINS
             for sd in stores.values() if eligible(sd, pd.Timestamp(o))]
    out = Parallel(n_jobs=4)(delayed(run)(*t) for t in tasks)
    rows = []
    for name, origin, store, point in out:
        actual = stores[store].monthly_actual()
        for month, f in point.items():
            if (month - origin.to_period("M")).n > SKIP_MONTHS:
                rows.append((name, origin, store, month, actual[month], f))
    df = pd.DataFrame(rows, columns=["variant", "origin", "store", "month", "actual", "forecast"])
    res = []
    for name, g in df.groupby("variant", sort=False):
        brand = g.groupby(["origin", "month"])[["actual", "forecast"]].sum().reset_index()
        win = [bias(b.actual, b.forecast) for _, b in brand.groupby("origin")]
        res.append({"variant": name, "store_month_wape": wape(g.actual, g.forecast),
                    "brand_month_mape": mape(brand.actual, brand.forecast),
                    "brand_window_error": np.mean(win), "brand_window_abs_error": np.mean(np.abs(win))})
    res = pd.DataFrame(res)
    res.to_csv(OUTPUTS / "prophet_sensitivity.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
