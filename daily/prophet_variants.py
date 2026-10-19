"""What if Prophet gets every special day (past and future) and its own weekly pattern for summer (and for the
holiday season)? Same backtest as daily/backtest.py: alone and blended with the tree model.

Extended holiday table = the repository's (US holidays, Valentine's, Easter, Mother's and Father's Day, Black Friday,
Christmas Eve, New Year's Eve) plus Dec 26, the days between Christmas and New Year's Eve, the Saturday and Sunday
after Thanksgiving, school spring break (the week either side of Easter) and Halloween.
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
from common import (DAYS_OFF, HOLIDAYS, LAST_DAY, OUT, PROPHET_CFG, STORES, season_of,  # noqa: E402
                    shape_errors)
from forecasting import load_stores  # noqa: E402


def extended_holidays(years=range(2022, 2029)) -> pd.DataFrame:
    rows = []
    for y in years:
        rows.append((pd.Timestamp(y, 12, 26), "Dec 26"))
        rows += [(pd.Timestamp(y, 12, d), "Holiday week") for d in range(27, 31)]
        rows.append((pd.Timestamp(y, 10, 31), "Halloween"))
        tg = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="W-THU")[3]
        rows += [(tg + pd.Timedelta(days=2), "Thanksgiving weekend"), (tg + pd.Timedelta(days=3), "Thanksgiving weekend")]
        easter = HOLIDAYS[(HOLIDAYS.holiday == "Easter") & (HOLIDAYS.ds.dt.year == y)].ds.iloc[0]
        rows += [(d, "Spring break") for d in pd.date_range(easter - pd.Timedelta(days=7), easter + pd.Timedelta(days=7))
                 if d != easter]
    extra = pd.DataFrame(rows, columns=["ds", "holiday"])
    extra = extra[~extra.ds.isin(HOLIDAYS.ds)]  # a named holiday keeps its own effect
    return pd.concat([HOLIDAYS, extra], ignore_index=True).sort_values("ds").reset_index(drop=True)


HOL_EXT = extended_holidays()
CONDS = {"summer": ["summer", "rest"], "season3": ["summer", "holseason", "rest"]}


def conditions(ds, kind):
    s = season_of(pd.DatetimeIndex(ds))
    if kind == "summer":
        return {"summer": s == "summer", "rest": s != "summer"}
    return {"summer": s == "summer", "holseason": s == "holidays", "rest": (s != "summer") & (s != "holidays")}


def fit(sd, origin, dates, weekly="tuned", holidays="base"):
    from prophet import Prophet

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    c = PROPHET_CFG
    m = Prophet(growth="linear", seasonality_mode=c["seasonality_mode"], yearly_seasonality=int(c["yearly_fourier"]),
                weekly_seasonality=weekly == "tuned", daily_seasonality=False,
                holidays=HOL_EXT if holidays == "ext" else HOLIDAYS,
                changepoint_prior_scale=c["changepoint_prior_scale"],
                seasonality_prior_scale=c["seasonality_prior_scale"],
                holidays_prior_scale=c["holidays_prior_scale"], changepoint_range=c["changepoint_range"],
                uncertainty_samples=0)
    train = sd.train_days(origin)
    fut = pd.DataFrame({"ds": dates})
    if weekly in CONDS:
        for name in CONDS[weekly]:
            m.add_seasonality(f"weekly_{name}", period=7, fourier_order=3, prior_scale=10.0, condition_name=name)
        for df in (train, fut):
            for k, v in conditions(df.ds, weekly).items():
                df[k] = v
    m.fit(train)
    f = 1 + m.predict(fut).multiplicative_terms.to_numpy()
    return np.clip(f, 0.05, None) * sd.open_flags(dates)


VARIANTS = {"p_hol": ("tuned", "ext"), "p_summer_hol": ("summer", "ext"), "p_season3_hol": ("season3", "ext")}


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
    for v in ["prophet_tuned", "prophet_summer"] + list(VARIANTS):
        days[f"blend_{v}"] = np.sqrt(days[v] * days.lgbm_hl180)
    days.to_pickle(OUT / "backtest_days.pkl")
    rows = []
    for m in ["prophet_tuned", "prophet_summer", *VARIANTS, "lgbm_hl180", "blend_prophet_tuned",
              "blend_prophet_summer", *[f"blend_{v}" for v in VARIANTS]]:
        z = pd.concat([shape_errors(g, m) for _, g in days.groupby("origin")])
        kind = np.where(z.special.isin(DAYS_OFF), "holiday", np.where(z.special != "", "other special",
                                                                     np.where(z.dow >= 5, "weekend", "weekday")))
        sm = z.season.to_numpy() == "summer"
        r = {"method": m, "all": z.err.sum() / z.sales.sum() * 100,
             "summer": z.err[sm].sum() / z.sales[sm].sum() * 100}
        for k in ("weekday", "weekend", "holiday", "other special"):
            r[k] = z.err[kind == k].sum() / z.sales[kind == k].sum() * 100
        rows.append(r)
    out = pd.DataFrame(rows).set_index("method").round(2)
    out.to_csv(OUT / "prophet_variants.csv")
    print(out.to_string())


if __name__ == "__main__":
    main()
