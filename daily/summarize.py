"""Summary of the backtest (after backtest.py, experiments.py and holiday_test.py): daily error by method, store and
kind of day, and the chart comparing the current daily split (Prophet as tuned) with the chosen blend.

    python daily/summarize.py   -> daily/outputs/backtest_summary.csv, daily/outputs/backtest_by_store.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "report"))
from common import DAYS_OFF, NAME, OUT, STORES, shape_errors  # noqa: E402

METHODS = {"flat": "Equal days", "dow52": "One weekday pattern (52 weeks)", "prophet_tuned": "Prophet as tuned (current)",
           "prophet_free": "Prophet, free weekly pattern", "prophet_summer": "Prophet, summer weekly pattern",
           "structural": "Components (weekday by season, holidays, time of year)", "lgbm_hl180": "Tree model (LightGBM)",
           "blend": "Blend: Prophet as tuned + tree model (chosen)",
           "blend_hol": "Blend, days-off holidays at their usual level"}

days = pd.read_pickle(OUT / "backtest_days.pkl")
rows = []
for m, label in METHODS.items():
    z = pd.concat([shape_errors(g, m) for _, g in days.groupby("origin")])
    kind = np.where(z.special.isin(DAYS_OFF), "days-off holiday", np.where(z.special != "", "other special day",
                                                                          np.where(z.dow >= 5, "weekend", "weekday")))
    def w(mask):
        return z.err[mask].sum() / z.sales[mask].sum() * 100
    row = {"method": m, "label": label, "all": w(np.ones(len(z), bool))}
    for k in ("weekday", "weekend", "days-off holiday", "other special day"):
        row[k] = w(kind == k)
    for s in STORES:
        row[NAME.get(s, s)] = w((z.store == s).to_numpy())
    rows.append(row)
summary = pd.DataFrame(rows)
summary.round(2).to_csv(OUT / "backtest_summary.csv", index=False)
print(summary.set_index("label").drop(columns="method").round(2).T.to_string())

import charts  # noqa: E402
from build_report import copy_assets, find_design_system  # noqa: E402

charts.setup(copy_assets(find_design_system(None)))
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white"})
s = summary.set_index("method")
labels = ["All stores", "Holidays (days off)"] + [NAME.get(x, x) for x in STORES]
cur = [s.loc["prophet_tuned", "all"], s.loc["prophet_tuned", "days-off holiday"]] + \
      [s.loc["prophet_tuned", NAME.get(x, x)] for x in STORES]
new = [s.loc["blend", "all"], s.loc["blend", "days-off holiday"]] + [s.loc["blend", NAME.get(x, x)] for x in STORES]
fig, ax = plt.subplots(figsize=(11.5, 6.2))
ys = np.arange(len(labels))[::-1].astype(float)
ys[:2] += 0.6  # a little air between the totals and the stores
for y, a, b, lab in zip(ys, cur, new, labels):
    ax.plot([a, b], [y, y], color=charts.G300, lw=1.4, zorder=1)
    ax.plot(a, y, marker="D", ms=7.5, color=charts.G700, mec="white", mew=1.5, ls="none", zorder=2)
    ax.plot(b, y, marker="o", ms=9, color=charts.PINK, mec="white", mew=1.5, ls="none", zorder=3)
    ax.text(max(a, b) + 0.25, y, f"{a:.1f}% → {b:.1f}%".replace("→", "to"), va="center", fontsize=8.5, color=charts.G700)
ax.set_yticks(ys, labels)
for t in ax.get_yticklabels()[:2]:
    t.set_fontweight(600)
ax.set_xlim(0, 19)
ax.set_xticks(range(0, 19, 3), [f"{v}%" for v in range(0, 19, 3)])
ax.grid(axis="x", color=charts.G100)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Daily error given the true month (out of sample): how far each day lands from the actual day", fontsize=9)
ax.legend([plt.Line2D([], [], marker="D", ms=7, color=charts.G700, mec="white", ls="none"),
           plt.Line2D([], [], marker="o", ms=8, color=charts.PINK, mec="white", ls="none")],
          ["Current daily split: Prophet as tuned", "New: Prophet as tuned + tree model, blended"], loc="lower left",
          bbox_to_anchor=(-0.005, 1.0), ncol=2, fontsize=9)
fig.text(0.012, 0.985, "The blended daily model splits each month into days with less error",
         ha="left", va="top", fontsize=15, fontweight=600, color=charts.INK)
fig.text(0.012, 0.935, "Backtest: six forecast origins (Dec 2024 – Mar 2026); each month of the following year split "
         "into days from data up to the origin and compared with the actual days.", ha="left", va="top", fontsize=9.5,
         color=charts.G700)
fig.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.1)
fig.savefig(OUT / "backtest_by_store.png", dpi=170)
print("chart ->", (OUT / "backtest_by_store.png").name)
