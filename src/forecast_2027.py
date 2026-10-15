"""Fit the three models on all history through Sep-2026 and forecast Oct-2026 .. Dec-2027.

Uses the settings adopted by src/tune.py (outputs/tuned_configs.json) when present, otherwise
the round-1 defaults. Alongside the best estimate (the blend) it writes the conservative "Plan"
= blend x the calibrated factor, because over-forecasting sales costs more than under-forecasting.

Outputs (outputs/):
  forecast_by_store_month.csv   point forecasts per store x month x model (+ actuals to date)
  forecast_brand_month.csv      brand totals per month: each model, ensemble, p10/p90 ranges
  forecast_annual.csv           2026 landing (actual Jan-Sep + forecast Oct-Dec) and 2027 by store
  october_check.csv             Oct 1-14 2026 forecast vs actual (out-of-sample live check)
  model_specs.csv               SARIMA order chosen per store
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest import ENSEMBLE
from forecasting import (FORECASTERS, MODELS, OUTPUTS, REGION, STORES, combine, load_configs,
                         load_daily, load_stores)

ORIGIN = pd.Timestamp("2026-09-30")
LAST_MONTH = pd.Period("2027-12", "M")
LIVE_CHECK_END = pd.Timestamp("2026-10-14")
N_SAMPLES = 1000
CONFIGS = load_configs()
PLAN = "Plan"  # conservative plan = blend x calibrated factor (see src/tune.py step 6)
PLAN_FACTOR = CONFIGS["plan"]["factor"]
SERIES = MODELS + [ENSEMBLE, PLAN]


def run(sd, model, seed):
    point, samples = FORECASTERS[model](sd, ORIGIN, LAST_MONTH, n_samples=N_SAMPLES, seed=seed,
                                        cfg=CONFIGS["models"][model])
    return sd.store, model, point, samples


def blend_paths(paths: dict) -> np.ndarray:
    """Apply the blend rule draw by draw to the models' simulated brand paths."""
    rule = CONFIGS["blend"]
    stack = np.stack([paths[m] for m in MODELS])
    if rule == "mean":
        return stack.mean(axis=0)
    if rule == "median":
        return np.median(stack, axis=0)
    w = np.array([rule[m] for m in MODELS])
    return np.tensordot(w / w.sum(), stack, axes=1)


def main() -> None:
    stores = load_stores(LIVE_CHECK_END.strftime("%Y-%m-%d"))
    tasks = [(stores[s], m, 1000 * i + j) for i, s in enumerate(STORES) for j, m in enumerate(MODELS)]
    out = Parallel(n_jobs=4)(delayed(run)(*t) for t in tasks)
    points = {(s, m): p for s, m, p, _ in out}
    samples = {(s, m): x for s, m, _, x in out}
    months = pd.period_range(ORIGIN.to_period("M") + 1, LAST_MONTH, freq="M")

    # ---- store x month point forecasts --------------------------------------------------
    rows = []
    for s in STORES:
        for month in months:
            r = {"store": s, "region": REGION[s], "month": str(month)}
            r.update({m: points[(s, m)][month] for m in MODELS})
            rows.append(r)
    by_store = pd.DataFrame(rows)
    by_store[ENSEMBLE] = combine(by_store, CONFIGS["blend"])
    by_store[PLAN] = by_store[ENSEMBLE] * PLAN_FACTOR
    by_store.to_csv(OUTPUTS / "forecast_by_store_month.csv", index=False)

    # ---- brand totals with simulation ranges ---------------------------------------------
    # Paths are summed across stores draw by draw (stores treated as independent).
    brand_paths = {m: sum(samples[(s, m)].to_numpy() for s in STORES) for m in MODELS}
    brand_paths[ENSEMBLE] = blend_paths(brand_paths)
    brand_paths[PLAN] = brand_paths[ENSEMBLE] * PLAN_FACTOR
    brand = pd.DataFrame({"month": [str(x) for x in months]})
    for m in SERIES:
        brand[m] = by_store.groupby("month")[m].sum().reindex(brand.month).to_numpy()
        brand[f"{m} p10"] = np.percentile(brand_paths[m], 10, axis=1)
        brand[f"{m} p90"] = np.percentile(brand_paths[m], 90, axis=1)
    brand.to_csv(OUTPUTS / "forecast_brand_month.csv", index=False)

    # ---- annual view: 2026 landing and 2027 ----------------------------------------------
    is27 = np.array([x.year == 2027 for x in months])
    q4_26 = np.array([x.year == 2026 for x in months])
    raw = load_daily()  # actuals include opening-ramp days the models skip
    raw_monthly = raw.groupby(["store", raw.date.dt.to_period("M")]).net_sales.sum()
    annual = []
    for s in STORES + ["TOTAL"]:
        act = raw_monthly.groupby(level=1).sum() if s == "TOTAL" else raw_monthly.loc[s]
        act = act.reindex(pd.period_range("2022-09", "2026-10", freq="M")).fillna(0)
        r = {"store": s, "region": REGION.get(s, "All"),
             "actual_2023": act[act.index.year == 2023].sum(),
             "actual_2024": act[act.index.year == 2024].sum(),
             "actual_2025": act[act.index.year == 2025].sum(),
             "actual_2026_jan_sep": act[(act.index.year == 2026) & (act.index.month <= 9)].sum()}
        sel = by_store if s == "TOTAL" else by_store[by_store.store == s]
        g = sel.groupby("month")[SERIES].sum()
        for m in SERIES:
            r[f"{m} 2026E"] = r["actual_2026_jan_sep"] + g[m].to_numpy()[q4_26].sum()
            r[f"{m} 2027"] = g[m].to_numpy()[is27].sum()
            if s == "TOTAL":
                tot27 = brand_paths[m][is27].sum(axis=0)
                r[f"{m} 2027 p10"], r[f"{m} 2027 p90"] = np.percentile(tot27, [10, 90])
        annual.append(r)
    annual = pd.DataFrame(annual)
    annual.to_csv(OUTPUTS / "forecast_annual.csv", index=False)

    # ---- live check: Oct 1-14 2026, prorating October by trading-day weights ---------------
    oct_days = pd.date_range("2026-10-01", "2026-10-31")
    live = []
    for s in STORES:
        sd = stores[s]
        w = sd.dow_weights(ORIGIN)[oct_days.dayofweek] * sd.open_flags(oct_days)
        share = w[oct_days <= LIVE_CHECK_END].sum() / w.sum()
        actual = sd.daily.loc["2026-10-01":LIVE_CHECK_END, "sales"].fillna(0).sum()
        r = {"store": s, "actual_oct_1_14": actual}
        for m in MODELS:
            r[m] = points[(s, m)][pd.Period("2026-10", "M")] * share
        live.append(r)
    live = pd.DataFrame(live)
    live[ENSEMBLE] = combine(live, CONFIGS["blend"])
    live[PLAN] = live[ENSEMBLE] * PLAN_FACTOR
    live.to_csv(OUTPUTS / "october_check.csv", index=False)

    pd.DataFrame([{"store": s, "sarima_spec": points[(s, "SARIMA")].attrs.get("spec")} for s in STORES]) \
        .to_csv(OUTPUTS / "model_specs.csv", index=False)

    # ---- console summary -------------------------------------------------------------------
    pd.set_option("display.width", 220)
    cols = ["store", "actual_2025", "actual_2026_jan_sep"] + [f"{m} 2026E" for m in [ENSEMBLE]] + \
           [f"{m} 2027" for m in SERIES]
    view = annual[cols].copy()
    for c in cols[1:]:
        view[c] = (view[c] / 1e6).round(2)
    print(view.to_string(index=False))
    t = annual[annual.store == "TOTAL"].iloc[0]
    for m in SERIES:
        print(f"{m:13s} 2026E {t[f'{m} 2026E']/1e6:6.2f}M  2027 {t[f'{m} 2027']/1e6:6.2f}M "
              f"({t[f'{m} 2027']/t[f'{m} 2026E']-1:+.1%})  p10-p90 {t[f'{m} 2027 p10']/1e6:.2f}-{t[f'{m} 2027 p90']/1e6:.2f}M")
    tl = live[["actual_oct_1_14"] + SERIES].sum()
    print("Oct 1-14 live check:", {k: round(v) for k, v in tl.items()},
          {m: f"{tl[m]/tl.actual_oct_1_14-1:+.1%}" for m in SERIES})


if __name__ == "__main__":
    main()
