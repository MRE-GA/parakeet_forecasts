"""Is there seasonality within the month (paydays, start or end of month) that improves the daily split?
Same backtest as daily/backtest.py, three ways, each learned only from data before the origin:
  1. Prophet with a monthly Fourier seasonality (period 30.44 days, order 3 or 5; moderate or free prior)
  2. the tree model with and without its day-of-month features (day of month, days left in the month)
  3. an explicit day-of-month adjustment on the blend: the pooled profile by day of month, net of each month's
     level, the store's weekday-by-season pattern and special days, shrunk toward zero
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
import common  # noqa: E402
from common import (LAST_DAY, LGBM, OUT, PROPHET_CFG, STORES, build_panel, shape_errors)  # noqa: E402
from forecasting import load_stores  # noqa: E402
from prophet_variants import HOL_EXT  # noqa: E402

MONTH_DAYS = 365.25 / 12


def fit(sd, origin, dates, order, prior):
    from prophet import Prophet

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    c = PROPHET_CFG
    m = Prophet(growth="linear", seasonality_mode=c["seasonality_mode"], yearly_seasonality=int(c["yearly_fourier"]),
                weekly_seasonality=True, daily_seasonality=False, holidays=HOL_EXT,
                changepoint_prior_scale=c["changepoint_prior_scale"],
                seasonality_prior_scale=c["seasonality_prior_scale"],
                holidays_prior_scale=c["holidays_prior_scale"], changepoint_range=c["changepoint_range"],
                uncertainty_samples=0)
    m.add_seasonality("monthly", period=MONTH_DAYS, fourier_order=order, prior_scale=prior)
    m.fit(sd.train_days(origin))
    f = 1 + m.predict(pd.DataFrame({"ds": dates})).multiplicative_terms.to_numpy()
    return np.clip(f, 0.05, None) * sd.open_flags(dates)


VARIANTS = {"p_hol_m3_mod": (3, 0.1), "p_hol_m5_mod": (5, 0.1), "p_hol_m5_free": (5, 10.0)}


def dom_profile(panel, origin, shrink=200.0):
    """Pooled log effect by day of month (1-31) from normal non-special days before the origin."""
    z = panel[panel.normal & (panel.special == "") & (panel.date <= origin)].copy()
    z["r"] = np.log(z.sales) - np.log(z.sales).groupby([z.store, z.month]).transform("mean")
    z["r"] -= z.groupby(["store", "season", "dow"]).r.transform("mean")
    z["r"] -= z.groupby(["store", "month"]).r.transform("mean")
    g = z.groupby("dom").r.agg(["sum", "count"]).reindex(range(1, 32), fill_value=0)
    return (g["sum"] / (g["count"] + shrink)).to_numpy()  # shrunk toward zero


def main() -> None:
    t0 = time.time()
    days = pd.read_pickle(OUT / "backtest_days.pkl")
    panel = build_panel()
    stores = load_stores(LAST_DAY)
    feats_full = list(common.LGB_FEATURES)
    for origin, idx in days.groupby("origin").groups.items():
        test = days.loc[idx]
        jobs = [(s, v) for s in STORES for v in VARIANTS if (test.store == s).any()]
        res = Parallel(n_jobs=4)(delayed(fit)(stores[s], origin, pd.DatetimeIndex(test[test.store == s].date),
                                              *VARIANTS[v]) for s, v in jobs)
        for (s, v), w in zip(jobs, res):
            days.loc[test.index[test.store == s], v] = w
        common.LGB_FEATURES = [f for f in feats_full if f not in ("dom", "days_left")]
        days.loc[idx, "lgbm_nodom"] = LGBM(panel, origin, half_life=180).weights(test)
        common.LGB_FEATURES = feats_full
        prof = dom_profile(panel, origin)
        days.loc[idx, "dom_adj"] = np.exp(prof[test.dom.to_numpy() - 1])
        print(f"origin {origin.date()}: {time.time() - t0:.0f}s", flush=True)
    days["blend_hol"] = np.sqrt(days.p_hol * days.lgbm_hl180)
    for v in VARIANTS:
        days[f"blend_{v}"] = np.sqrt(days[v] * days.lgbm_hl180)
    days["blend_hol_nodom"] = np.sqrt(days.p_hol * days.lgbm_nodom)
    days["blend_hol_domadj"] = days.blend_hol * days.dom_adj
    days.to_pickle(OUT / "backtest_days.pkl")
    rows = []
    for m in ["p_hol", *VARIANTS, "lgbm_hl180", "lgbm_nodom", "blend_hol", *[f"blend_{v}" for v in VARIANTS],
              "blend_hol_nodom", "blend_hol_domadj"]:
        z = pd.concat([shape_errors(g, m) for _, g in days.groupby("origin")])
        dim = pd.DatetimeIndex(z.date).days_in_month
        part = np.select([z.dom <= 3, z.dom.between(14, 16), z.dom >= dim - 2], ["start", "mid", "end"], "rest")
        r = {"method": m, "all": z.err.sum() / z.sales.sum() * 100}
        for k in ("start", "mid", "end", "rest"):
            r[f"days_{k}"] = z.err[part == k].sum() / z.sales[part == k].sum() * 100
        rows.append(r)
    out = pd.DataFrame(rows).set_index("method").round(2)
    out.to_csv(OUT / "intramonth.csv")
    print(out.to_string())


if __name__ == "__main__":
    main()
