"""Variants on the best methods, on the same backtest (daily/backtest.py must have run): more weight on recent
weeks for the tree model and the weekday pattern, and different blends with Prophet."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LGBM, OUT, Structural, build_panel, shape_errors  # noqa: E402

days = pd.read_pickle(OUT / "backtest_days.pkl")
panel = build_panel()
t0 = time.time()
for origin, idx in days.groupby("origin").groups.items():
    test = days.loc[idx]
    for hl in (90, 180):
        days.loc[idx, f"lgbm_hl{hl}"] = LGBM(panel, origin, half_life=hl).weights(test)
    print(origin.date(), f"{time.time() - t0:.0f}s", flush=True)
days["ens_pt_l180"] = np.sqrt(days.prophet_tuned * days.lgbm_hl180)
days["ens_pt_l90"] = np.sqrt(days.prophet_tuned * days.lgbm_hl90)
days["ens_pt_l"] = np.sqrt(days.prophet_tuned * days.lgbm)
for w in (0.3, 0.4, 0.6, 0.7):
    days[f"ens_w{int(w * 10)}"] = days.prophet_tuned ** (1 - w) * days.lgbm_hl180 ** w
methods = ["prophet_tuned", "lgbm", "lgbm_hl180", "lgbm_hl90", "ens_pt_l", "ens_pt_l180", "ens_pt_l90", "ens_w3",
           "ens_w4", "ens_w6", "ens_w7"]
res = {}
for m in methods:
    z = pd.concat([shape_errors(g, m) for _, g in days.groupby("origin")])
    z["sgn"] = z.pred - z.sales
    g = z.groupby("dow")[["sgn", "sales"]].sum()
    res[m] = {"wape": z.err.sum() / z.sales.sum() * 100,
              **{f"bias_{d}": v for d, v in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                                 (g.sgn / g.sales * 100).round(2))}}
print(pd.DataFrame(res).T.round(2).to_string())
days.to_pickle(OUT / "backtest_days.pkl")
