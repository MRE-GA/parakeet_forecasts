"""Does setting days-off holidays to their usual level (whatever the weekday) improve the daily split? Same
backtest as daily/backtest.py; the holiday ratios are learned only from occurrences before each origin."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAYS_OFF, OUT, build_panel, holiday_override, holiday_ratios, shape_errors  # noqa: E402

days = pd.read_pickle(OUT / "backtest_days.pkl")
panel = build_panel()
days["blend"] = np.sqrt(days.prophet_tuned * days.lgbm_hl180)
days["blend_hol"] = days.blend
for origin, idx in days.groupby("origin").groups.items():
    test = days.loc[idx]
    days.loc[idx, "blend_hol"] = holiday_override(test, test.blend.to_numpy(), holiday_ratios(panel, origin))
for m in ("prophet_tuned", "blend", "blend_hol"):
    z = pd.concat([shape_errors(g, m) for _, g in days.groupby("origin")])
    sp = z[z.special.isin(DAYS_OFF)]
    wk = sp[sp.dow >= 5]
    print(f"{m:14s} all days {z.err.sum() / z.sales.sum() * 100:5.2f}%   days-off holidays "
          f"{sp.err.sum() / sp.sales.sum() * 100:5.2f}% (n={len(sp)})   on weekends {wk.err.sum() / wk.sales.sum() * 100:5.2f}% "
          f"(n={len(wk)})")
    print("   by holiday:", (sp.groupby("special").apply(lambda g: g.err.sum() / g.sales.sum() * 100)).round(1).to_dict())
days.to_pickle(OUT / "backtest_days.pkl")
