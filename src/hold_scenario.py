"""'No growth after September' scenario for 2027, per café and month.

Prophet (with the adopted settings) is fitted on each café's history through Sep 30, 2026 like the
2027 forecast, but its trend is frozen at its Sep 30 value: 2027 keeps the café's seasonal and
holiday pattern at today's level, with no further growth. The budget review compares the
investor budget with this path.

    python src/hold_scenario.py      -> outputs/hold_scenario_2027.csv
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecasting import HOLIDAYS, OUTPUTS, PROPHET_DEFAULT, STORES, load_configs, load_stores

ORIGIN = pd.Timestamp("2026-09-30")
CFG = {**PROPHET_DEFAULT, **load_configs()["models"]["Prophet"]}
OUT = OUTPUTS / "hold_scenario_2027.csv"


def frozen_trend(sd) -> pd.DataFrame:
    from prophet import Prophet

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    c = CFG
    if c["summer_weekly"] or c["holiday_windows"] or c["log_y"] or c["cap_outliers"]:
        raise NotImplementedError("hold scenario assumes the plain Prophet feature set")
    m = Prophet(growth="linear", seasonality_mode=c["seasonality_mode"], yearly_seasonality=int(c["yearly_fourier"]),
                weekly_seasonality=True, daily_seasonality=False, holidays=HOLIDAYS,
                changepoint_prior_scale=c["changepoint_prior_scale"],
                seasonality_prior_scale=c["seasonality_prior_scale"],
                holidays_prior_scale=c["holidays_prior_scale"], changepoint_range=c["changepoint_range"],
                uncertainty_samples=0)
    m.fit(sd.train_days(ORIGIN))
    future = pd.DataFrame({"ds": pd.date_range("2027-01-01", "2027-12-31")})
    p = m.predict(future)
    level = m.predict(pd.DataFrame({"ds": [ORIGIN]})).trend.iloc[0]
    daily = (level * (1 + p.multiplicative_terms) + p.additive_terms).clip(lower=0).to_numpy()
    daily = daily * sd.open_flags(future.ds)
    monthly = pd.Series(daily, index=future.ds.dt.to_period("M")).groupby(level=0).sum()
    return pd.DataFrame({"store": sd.store, "month": monthly.index.astype(str), "hold": monthly.to_numpy()})


def main() -> None:
    stores = load_stores("2026-10-14")
    out = Parallel(n_jobs=4)(delayed(frozen_trend)(stores[s]) for s in STORES)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT, index=False)
    print(df.groupby("store").hold.sum().round(0).to_string())
    print(f"-> {OUT.relative_to(OUTPUTS.parent)}")


if __name__ == "__main__":
    main()
