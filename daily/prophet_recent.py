"""Prophet with a weekly pattern of its own for recent data (conditional seasonality by period): the last 12 (or 6)
months before the origin get one weekly pattern, earlier data another, and the forecast uses the recent one. The rest
of the history still sets the time of year and the special days. Same backtest as daily/backtest.py, on top of the
extended special days (prophet_variants.py); alone and blended with the tree model.
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAYS_OFF, LAST_DAY, OUT, PROPHET_CFG, STORES, season_of, shape_errors  # noqa: E402
from forecasting import load_stores  # noqa: E402
from prophet_variants import HOL_EXT  # noqa: E402

TUNED = PROPHET_CFG["seasonality_prior_scale"]
# name: (days counted as recent, prior scale of the weekly patterns, split by summer as well)
VARIANTS = {"p_rec365_free": (365, 10.0, False), "p_rec365_tuned": (365, TUNED, False),
            "p_rec182_free": (182, 10.0, False), "p_rec365_summer_free": (365, 10.0, True)}


def conditions(ds, origin, days, by_summer, future):
    ds = pd.DatetimeIndex(ds)
    recent = np.ones(len(ds), bool) if future else np.asarray(ds > origin - pd.Timedelta(days=days))
    if not by_summer:
        return {"recent": recent, "older": ~recent}
    summer = season_of(ds) == "summer"
    return {"recent_summer": recent & summer, "recent_rest": recent & ~summer,
            "older_summer": ~recent & summer, "older_rest": ~recent & ~summer}


def fit(sd, origin, dates, days, prior, by_summer):
    from prophet import Prophet

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    c = PROPHET_CFG
    m = Prophet(growth="linear", seasonality_mode=c["seasonality_mode"], yearly_seasonality=int(c["yearly_fourier"]),
                weekly_seasonality=False, daily_seasonality=False, holidays=HOL_EXT,
                changepoint_prior_scale=c["changepoint_prior_scale"],
                seasonality_prior_scale=c["seasonality_prior_scale"],
                holidays_prior_scale=c["holidays_prior_scale"], changepoint_range=c["changepoint_range"],
                uncertainty_samples=0)
    train = sd.train_days(origin)
    fut = pd.DataFrame({"ds": dates})
    for k, v in conditions(train.ds, origin, days, by_summer, False).items():
        train[k] = v
    for k, v in conditions(fut.ds, origin, days, by_summer, True).items():
        fut[k] = v
    for k in conditions(fut.ds, origin, days, by_summer, True):
        if train[k].sum() >= 28:  # a condition needs data to learn from
            m.add_seasonality(f"weekly_{k}", period=7, fourier_order=3, prior_scale=prior, condition_name=k)
    m.fit(train)
    f = 1 + m.predict(fut).multiplicative_terms.to_numpy()
    return np.clip(f, 0.05, None) * sd.open_flags(dates)


def main() -> None:
    t0 = time.time()
    days = pd.read_pickle(OUT / "backtest_days.pkl")
    stores = load_stores(LAST_DAY)
    for origin, idx in days.groupby("origin").groups.items():
        test = days.loc[idx]
        jobs = [(s, v) for s in STORES for v in VARIANTS if (test.store == s).any()]
        res = Parallel(n_jobs=4)(delayed(fit)(stores[s], origin, pd.DatetimeIndex(test[test.store == s].date),
                                              *VARIANTS[v]) for s, v in jobs)
        for (s, v), w in zip(jobs, res):
            days.loc[test.index[test.store == s], v] = w
        print(f"origin {origin.date()}: {time.time() - t0:.0f}s", flush=True)
    for v in VARIANTS:
        days[f"blend_{v}"] = np.sqrt(days[v] * days.lgbm_hl180)
    days.to_pickle(OUT / "backtest_days.pkl")
    rows = []
    order = ["prophet_tuned", "p_hol", "p_season3_hol", *VARIANTS, "lgbm_hl180", "blend_prophet_tuned", "blend_p_hol",
             "blend_p_season3_hol", *[f"blend_{v}" for v in VARIANTS]]
    for m in order:
        z = pd.concat([shape_errors(g, m) for _, g in days.groupby("origin")])
        kind = np.where(z.special.isin(DAYS_OFF), "holiday", np.where(z.special != "", "other special",
                                                                     np.where(z.dow >= 5, "weekend", "weekday")))
        z["sgn"] = z.pred - z.sales
        g = z.groupby("dow")[["sgn", "sales"]].sum()
        bias = g.sgn / g.sales * 100
        r = {"method": m, "all": z.err.sum() / z.sales.sum() * 100}
        for k in ("weekday", "weekend", "holiday"):
            r[k] = z.err[kind == k].sum() / z.sales[kind == k].sum() * 100
        r["max_weekday_bias"] = bias.abs().max()
        for s in ("Beverly Hills", "La Jolla Cove", "Newport", "Coronado", "Brentwood"):
            q = z[z.store == s]
            r[s] = q.err.sum() / q.sales.sum() * 100
        rows.append(r)
    out = pd.DataFrame(rows).set_index("method").round(2)
    out.to_csv(OUT / "prophet_recent.csv")
    print(out.to_string())


if __name__ == "__main__":
    main()
