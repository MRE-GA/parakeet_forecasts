"""2026 finish and 2027 at today's deseasonalized run-rate, per café: no growth from here, normal seasons.

Primary method (daily): Prophet, with the adopted settings, is fitted on each café's history through
Sep 30, 2026 and used only for its seasonal pattern (weekly, yearly and holidays). Dividing the last
45 days of actual sales by those seasonal factors gives the café's underlying daily level; 2027 is
that level multiplied back by the 2027 factors on the days the café trades. The 2026 finish is the
actual sales to Oct 14 plus the same level, seasonally adjusted, for Oct 15 - Dec 31.

Cross-check (no model): the 12-month centred average of monthly sales around September 2025 is free
of seasonality; lifting it by the café's growth over the last 45 days against the same weekdays a
year earlier gives the same underlying level today, held for 12 months.

    python src/runrate_scenario.py      -> outputs/runrate_45d_2027.csv
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecasting import HOLIDAYS, OUTPUTS, PROPHET_DEFAULT, ROOT, STORES, load_configs, load_daily, load_stores

ORIGIN = pd.Timestamp("2026-09-30")  # Prophet is fitted through here, like the 2027 forecast
WINDOW = pd.date_range("2026-08-31", "2026-10-14")  # last 45 days of Toast data
LAG = pd.Timedelta(days=364)  # same weekdays a year earlier
YEAR = pd.date_range("2027-01-01", "2027-12-31")
REST = pd.date_range("2026-10-15", "2026-12-31")  # rest of 2026 after the last day of data
CFG = {**PROPHET_DEFAULT, **load_configs()["models"]["Prophet"]}
OUT = OUTPUTS / "runrate_45d_2027.csv"
BUDGET = ROOT / "data" / "cfo_budget_2027.csv"


def seasonal_runrate(sd, window_sales: np.ndarray) -> dict:
    from prophet import Prophet

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    c = CFG
    if c["summer_weekly"] or c["holiday_windows"] or c["log_y"] or c["cap_outliers"]:
        raise NotImplementedError("run-rate scenario assumes the plain Prophet feature set")
    m = Prophet(growth="linear", seasonality_mode=c["seasonality_mode"], yearly_seasonality=int(c["yearly_fourier"]),
                weekly_seasonality=True, daily_seasonality=False, holidays=HOLIDAYS,
                changepoint_prior_scale=c["changepoint_prior_scale"],
                seasonality_prior_scale=c["seasonality_prior_scale"],
                holidays_prior_scale=c["holidays_prior_scale"], changepoint_range=c["changepoint_range"],
                uncertainty_samples=0)
    m.fit(sd.train_days(ORIGIN))

    def factors(dates: pd.DatetimeIndex) -> np.ndarray:
        p = m.predict(pd.DataFrame({"ds": dates}))
        return (1 + p.multiplicative_terms.to_numpy()) * sd.open_flags(dates)

    f_window, f_rest, f_year = factors(WINDOW), factors(REST), factors(YEAR)
    level = window_sales.sum() / f_window.sum()  # underlying sales per normal trading day
    days = REST.append(YEAR)
    monthly = pd.Series(level * np.r_[f_rest, f_year], index=days.to_period("M")).groupby(level=0).sum()
    return {"store": sd.store, "level_per_day": level, "runrate_2027": level * f_year.sum(),
            "runrate_rest_2026": level * f_rest.sum(), "monthly": monthly}


def main() -> None:
    raw = load_daily()
    stores = load_stores("2026-10-14")
    x = raw.set_index(["store", "date"]).net_sales
    window = {s: np.array([x.get((s, d), 0.0) for d in WINDOW]) for s in STORES}
    out = Parallel(n_jobs=4)(delayed(seasonal_runrate)(stores[s], window[s]) for s in STORES)

    budget = pd.read_csv(BUDGET).dropna(subset=["store"]).set_index("store")
    annual = pd.read_csv(OUTPUTS / "forecast_annual.csv").set_index("store")
    rows = []
    for r in out:
        s = r["store"]
        cur = window[s]
        prev = np.array([x.get((s, d - LAG), 0.0) for d in WINDOW])
        both = (cur > 0) & (prev > 0)  # days both years traded
        growth = cur[both].sum() / prev[both].sum() - 1
        monthly = stores[s].monthly_clean(ORIGIN)
        centred = monthly.rolling(12).mean().rolling(2).mean().shift(-6)  # 2x12 centred moving average
        base = 12 * centred[pd.Period("2025-09", "M")]
        to_date = raw[(raw.store == s) & raw.date.between("2026-01-01", WINDOW[-1])].net_sales.sum()
        rows.append({
            "store": s, "cafe": budget.budget_name.get(s, s),
            "actual_2026_to_oct14": to_date, "runrate_rest_2026": r["runrate_rest_2026"],
            "runrate_2026E": to_date + r["runrate_rest_2026"],
            "sales_last_45_days": cur.sum(), "growth_last_45_days": growth,
            "runrate_2027": r["runrate_2027"], "check_runrate_2027": base * (1 + growth),
            "our_2026_finish": annual.loc[s, "Average of 3 2026E"], "budget_2026_finish": budget.est_2026.get(s),
            "budget_2027": budget.objective_2027.get(s), "our_plan_2027": annual.loc[s, "Plan 2027"],
        })
    df = pd.DataFrame(rows).set_index("store").reindex(
        [s for s in budget.index] + [s for s in STORES if s not in budget.index])
    df["runrate_vs_2026"] = df.runrate_2027 / df.our_2026_finish - 1
    df["runrate_2027_vs_runrate_2026E"] = df.runrate_2027 / df.runrate_2026E - 1
    df["runrate_minus_budget"] = df.runrate_2027 - df.budget_2027
    df.to_csv(OUT, float_format="%.4f")
    pd.concat({r["store"]: r["monthly"] for r in out}, axis=1).T.to_csv(OUT.with_name("runrate_45d_2027_by_month.csv"),
                                                                          float_format="%.0f")
    show = df[["cafe", "growth_last_45_days", "runrate_2026E", "budget_2026_finish", "runrate_2027", "check_runrate_2027",
               "budget_2027",
               "runrate_minus_budget", "our_plan_2027", "runrate_vs_2026"]].copy()
    for k in ["runrate_2026E", "budget_2026_finish", "runrate_2027", "check_runrate_2027", "budget_2027",
              "runrate_minus_budget", "our_plan_2027"]:
        show[k] = (show[k] / 1e6).round(3)
    for k in ["growth_last_45_days", "runrate_vs_2026"]:
        show[k] = (show[k] * 100).round(1)
    print(show.to_string())
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
