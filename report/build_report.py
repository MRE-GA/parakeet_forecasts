"""Build the Parakeet-branded PDF report from the forecast outputs.

    python report/build_report.py [--ds PATH_TO_PARAKEET_DESIGN_SYSTEM]

Run src/backtest.py, src/tune.py and src/forecast_2027.py first.
Brand fonts, logo and photography come from the Parakeet Cafe design-system skill (not
committed to this repo); pass its folder with --ds or the PARAKEET_DS environment variable.
The PDF is printed by headless Chromium through Playwright (report/render_pdf.mjs).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report"))

import charts  # noqa: E402
from backtest import ENSEMBLE  # noqa: E402
from forecasting import (ANOMALIES, DEFAULT_CONFIGS, MODELS, OUTPUTS, REGION, STORE_START, STORES,  # noqa: E402
                         TUNED_CONFIGS, load_daily, load_stores)

BUILD = OUTPUTS / "report"
PDF = OUTPUTS / "Parakeet_2027_Sales_Forecast.pdf"
BLEND = "Blend (average of 3)"
LABEL = {**{m: m for m in MODELS}, ENSEMBLE: BLEND}
FONT_FILES = ["Monarcha-Bold.otf", "PlayfairDisplay-400.woff2", "PlayfairDisplay-400i.woff2",
              "PlayfairDisplay-700.woff2", "PlayfairDisplay-900.woff2", "JosefinSans-400.woff2",
              "JosefinSans-500.woff2", "JosefinSans-600.woff2", "JosefinSans-700.woff2"]
IMAGES = ["logo-bird.png", "parrot-map.png", "photos/hero-2.jpg"]


def find_design_system(arg: str | None) -> Path:
    for candidate in [arg, os.environ.get("PARAKEET_DS")]:
        if candidate and (Path(candidate) / "styles.css").exists():
            return Path(candidate)
    for p in (Path.home() / ".claude" / "skills").rglob("parakeet-cafe-design"):
        if (p / "styles.css").exists():
            return p
    sys.exit("Parakeet design system not found: pass --ds or set PARAKEET_DS")


def copy_assets(ds: Path) -> list[str]:
    assets = BUILD / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for f in FONT_FILES:
        shutil.copy(ds / "fonts" / f, assets / f)
    for f in IMAGES:
        shutil.copy(ds / "assets" / f, assets / Path(f).name)
    # matplotlib needs static TrueType instances of the (variable) chart font to measure text
    ttf = []
    for weight, style in ((400, "Regular"), (600, "SemiBold")):
        font = TTFont(assets / f"JosefinSans-{weight}.woff2")
        font.flavor = None
        if "fvar" in font:
            font = instancer.instantiateVariableFont(font, {"wght": weight})
        names = font["name"]
        names.names = [r for r in names.names if r.nameID not in (1, 2, 4, 6, 16, 17)]
        for name_id, value in ((1, "Josefin Sans"), (2, style), (4, f"Josefin Sans {style}"),
                               (6, f"JosefinSans-{style}")):
            names.setName(value, name_id, 3, 1, 0x409)
        font["OS/2"].usWeightClass = weight
        out = assets / f"JosefinSans-{style}.ttf"
        font.save(out)
        ttf.append(str(out))
    return ttf


# ------------------------------------------------------------------------------------------
# Numbers
# ------------------------------------------------------------------------------------------
def fm(x: float, d: int = 1) -> str:
    return f"${x / 1e6:,.{d}f}M"


def fk(x: float) -> str:
    return f"${x / 1e3:,.0f}K"


def pct(x: float, d: int = 0, sign: bool = True) -> str:
    return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"


def like_for_like(raw: pd.DataFrame) -> list[dict]:
    """Jan-Sep sales, checks and average ticket for cafes trading both full periods."""
    raw = raw[raw.date.dt.month <= 9].assign(year=raw.date.dt.year)
    rows = []
    for year in (2024, 2025, 2026):
        cafes = [s for s in STORES if pd.Timestamp(STORE_START[s]) <= pd.Timestamp(year - 1, 1, 1)]
        g = raw[raw.store.isin(cafes)].groupby("year")[["net_sales", "checks"]].sum()
        cur, prev = g.loc[year], g.loc[year - 1]
        rows.append({
            "period": f"Jan–Sep {year} vs {year - 1}", "cafes": len(cafes),
            "sales": cur.net_sales / prev.net_sales - 1,
            "checks": cur.checks / prev.checks - 1,
            "ticket": (cur.net_sales / cur.checks) / (prev.net_sales / prev.checks) - 1,
            "avg_ticket": cur.net_sales / cur.checks,
        })
    return rows


def holiday_lift(raw: pd.DataFrame) -> dict[str, float]:
    """Median sales on a holiday vs the cafe's surrounding 29-day median (all-week cafes)."""
    from forecasting import HOLIDAYS

    d = raw[raw.store != "Nautilus"].sort_values(["store", "date"]).copy()
    d["ratio"] = d.net_sales / d.groupby("store").net_sales.transform(
        lambda s: s.rolling(29, center=True, min_periods=10).median())
    d = d.merge(HOLIDAYS, left_on="date", right_on="ds")
    return d.groupby("holiday").ratio.median().sub(1).to_dict()


def collect() -> dict:
    raw = load_daily()
    annual = pd.read_csv(OUTPUTS / "forecast_annual.csv").set_index("store")
    brand_fc = pd.read_csv(OUTPUTS / "forecast_brand_month.csv")
    by_store = pd.read_csv(OUTPUTS / "forecast_by_store_month.csv")
    metrics = pd.read_csv(OUTPUTS / "backtest_metrics.csv")
    bt = pd.read_csv(OUTPUTS / "backtest_forecasts.csv", parse_dates=["origin"])
    live = pd.read_csv(OUTPUTS / "october_check.csv")
    specs = pd.read_csv(OUTPUTS / "model_specs.csv")
    all_models = MODELS + [ENSEMBLE]

    t = annual.loc["TOTAL"]
    n = {
        "first_day": raw.date.min(), "last_day": raw.date.max(), "store_days": len(raw),
        "total_sales": raw.net_sales.sum(),
        "a2023": t.actual_2023, "a2024": t.actual_2024, "a2025": t.actual_2025,
        "ytd2026": t.actual_2026_jan_sep, "e2026": t[f"{ENSEMBLE} 2026E"],
        "f2027": {m: t[f"{m} 2027"] for m in all_models},
    }
    n["g2026"] = n["e2026"] / n["a2025"] - 1
    n["g2027"] = {m: v / n["e2026"] - 1 for m, v in n["f2027"].items()}
    n["low"], n["high"] = min(n["f2027"][m] for m in MODELS), max(n["f2027"][m] for m in MODELS)
    n["low_model"] = min(MODELS, key=lambda m: n["f2027"][m])
    n["high_model"] = max(MODELS, key=lambda m: n["f2027"][m])
    n["slow_case"] = n["e2026"] * 1.08

    pooled = metrics[metrics.origin == "ALL"].set_index("model")
    folds = metrics[metrics.origin != "ALL"]
    n["bt"] = [{
        "model": LABEL[m], "key": m,
        "wape": pooled.loc[m, "store_month_wape"], "mape": pooled.loc[m, "brand_month_mape"],
        "abs_err": pooled.loc[m, "brand_window_abs_error"], "bias": pooled.loc[m, "brand_window_error"],
        "max_err": folds[folds.model == m].brand_window_error.abs().max(),
    } for m in all_models]
    singles = [r for r in n["bt"] if r["key"] in MODELS]
    n["best_single"] = min(singles, key=lambda r: r["wape"])["model"]
    n["bt_folds"] = folds.pivot(index="origin", columns="model", values="brand_window_error")[all_models]
    n["bt_fold_meta"] = folds.groupby("origin")[["stores", "months"]].first()
    n["bt_store_months"] = len(bt)
    n["n_folds"] = bt.origin.nunique()

    live_cols = all_models + (["Plan"] if "Plan" in live else [])
    lv = live[["actual_oct_1_14"] + live_cols].sum()
    n["live"] = {"actual": lv.actual_oct_1_14, **{m: lv[m] / lv.actual_oct_1_14 - 1 for m in live_cols}}

    n["lfl"] = like_for_like(raw)
    n["holiday"] = holiday_lift(raw)

    # new-cafe analogs: first 12 full months of sales
    analogs = []
    raw_m = raw.groupby(["store", raw.date.dt.to_period("M")]).net_sales.sum()
    for s in ["Newport", "Beverly Hills", "Brentwood", "Carlsbad"]:
        start = pd.Timestamp(STORE_START[s])
        first = start.to_period("M") if start.day == 1 else start.to_period("M") + 1
        analogs.append({"store": s, "opened": raw[raw.store == s].date.min(),
                        "first12": raw_m.loc[s].loc[first:first + 11].sum()})
    n["analogs"] = analogs
    n["analog_avg"] = np.mean([a["first12"] for a in analogs])

    # cafe table
    cafes = []
    for s in STORES:
        r = annual.loc[s]
        f = {m: r[f"{m} 2027"] for m in all_models}
        cafes.append({
            "store": s, "region": REGION[s], "a2025": r.actual_2025, "e2026": r[f"{ENSEMBLE} 2026E"],
            "f2027": f[ENSEMBLE], "growth": f[ENSEMBLE] / r[f"{ENSEMBLE} 2026E"] - 1,
            "low": min(f[m] for m in MODELS), "high": max(f[m] for m in MODELS),
            "prophet": f["Prophet"], "sarima": f["SARIMA"], "hw": f["Holt-Winters"],
            "share": f[ENSEMBLE] / n["f2027"][ENSEMBLE], "sarima_spec": specs.set_index("store").loc[s, "sarima_spec"],
            "plan": r["Plan 2027"] if "Plan 2027" in r.index else f[ENSEMBLE],
        })
        cafes[-1]["plan_growth"] = cafes[-1]["plan"] / cafes[-1]["e2026"] - 1
    cafes.sort(key=lambda c: -c["f2027"])
    n["cafes"] = cafes
    top = sorted(cafes, key=lambda c: -(c["f2027"] - c["e2026"]))
    n["growth_total"] = n["f2027"][ENSEMBLE] - n["e2026"]
    n["top3_share"] = sum(c["f2027"] - c["e2026"] for c in top[:3]) / n["growth_total"]
    n["top3"] = [c["store"] for c in top[:3]]
    n["slowest"] = sorted(cafes, key=lambda c: c["growth"])[:2]

    # brand months
    fc = brand_fc.assign(month=pd.PeriodIndex(brand_fc.month, freq="M"))
    fc["low"] = fc[MODELS].min(axis=1)
    fc["high"] = fc[MODELS].max(axis=1)
    raw_brand = raw.groupby(raw.date.dt.to_period("M")).net_sales.sum()
    # 2026 reference for each 2027 month: actual Jan-Sep, blended estimate Oct-Dec
    ref26 = {p: raw_brand[p] if p.month <= 9 else fc.set_index("month").loc[p, ENSEMBLE]
             for p in pd.period_range("2026-01", "2026-12", freq="M")}
    n["months27"] = [{"month": r.month.strftime("%b"), **{m: r[m] for m in all_models},
                      "plan": r["Plan"] if "Plan" in fc else r[ENSEMBLE], "ref2026": ref26[r.month - 12]}
                     for _, r in fc[fc.month.dt.year == 2027].iterrows()]
    headline = "plan" if "Plan" in fc else ENSEMBLE
    growth = [r[headline] / r["ref2026"] - 1 for r in n["months27"]]
    n["m_growth_min"], n["m_growth_max"] = min(growth), max(growth)
    q1 = n["months27"][:3]
    n["q1_growth"] = sum(r[headline] for r in q1) / sum(r["ref2026"] for r in q1) - 1
    # sensitivity: the two biggest growth contributors grow +10% instead
    fast = top[:2]
    n["fast2"] = fast
    n["fast2_case"] = n["f2027"][ENSEMBLE] - sum(c["f2027"] - c["e2026"] * 1.10 for c in fast)
    # half-width of each model's own 80% interval for the 2027 total
    n["pi_halfwidth"] = {m: (t[f"{m} 2027 p90"] - t[f"{m} 2027 p10"]) / 2 / t[f"{m} 2027"] for m in all_models}
    # illustrative: an average new cafe trading from July 1 (half of its first-year sales)
    n["new_cafe_half_year"] = n["analog_avg"] / 2
    n["fc"] = fc
    n["raw_brand"] = raw_brand
    n["by_store"] = by_store
    n["bt_df"] = bt
    n["anomalies"] = [f"{who or 'All cafés'}: {date_span(a, b)} ({why})" for who, a, b, why in ANOMALIES]
    n["fold_label"] = {o: pd.Timestamp(o).strftime("%b %-d, %Y") for o in n["bt_folds"].index}
    n["tuning"] = collect_tuning() if TUNED_CONFIGS.exists() else None
    n["plan"] = collect_plan(n, annual, by_store) if (TUNING / "plan_calibration.csv").exists() else None
    return n


def collect_plan(n: dict, annual: pd.DataFrame, by_store: pd.DataFrame) -> dict:
    """Conservative plan: the blend times a factor calibrated on an asymmetric cost."""
    cfg = json.loads(TUNED_CONFIGS.read_text())["plan"]
    cal = pd.read_csv(TUNING / "plan_calibration.csv")
    t = annual.loc["TOTAL"]
    blend27 = t[f"{ENSEMBLE} 2027"]
    rows = []
    for r in cal.itertuples():
        rows.append({"over_cost": int(r.over_cost), "factor_tuning": r.factor_tuning, "factor": r.factor_final,
                     "cost_blend": r.holdout_cost_blend, "cost_plan": r.holdout_cost_plan,
                     "over_blend": r.holdout_over_share_blend, "over_plan": r.holdout_over_share_plan,
                     "bias_plan": r.holdout_bias_plan, "f2027": blend27 * r.factor_final,
                     "chosen": int(r.over_cost) == int(cfg["over_cost"])})
    chosen = next(r for r in rows if r["chosen"])
    by_month = by_store.groupby("month")[["Plan"]].sum().Plan if "Plan" in by_store else None
    # cafe-level and horizon detail from the adopted blend's tuning + holdout forecasts
    from tune import best_factor
    adf = pd.read_csv(TUNING / "adopted_forecasts.csv")
    c = int(cfg["over_cost"])
    b = adf.groupby(["set", "origin", "month", "horizon"])[["actual", "f"]].sum().reset_index()
    hold = b[b.set == "holdout"]
    return {"over_cost": c, "factor": cfg["factor"], "f2027": t["Plan 2027"],
            "growth": t["Plan 2027"] / n["e2026"] - 1, "rows": rows, "chosen": chosen,
            "gap": t["Plan 2027"] - blend27, "by_month": by_month,
            "cafe_factor": best_factor(adf.actual, adf.f, c),
            "near_factor": best_factor(b[b.horizon <= 9].actual, b[b.horizon <= 9].f, c),
            "far_factor": best_factor(b[b.horizon >= 10].actual, b[b.horizon >= 10].f, c),
            "holdout_bias_blend": hold.f.sum() / hold.actual.sum() - 1,
            "holdout_months": len(hold)} | plan_narrative(rows, chosen, c, cfg["factor"], adf, b, hold, best_factor)


def plan_narrative(rows, chosen, c, factor, adf, b, hold, best_factor) -> dict:
    p1 = lambda x: pct(x, 1)  # noqa: E731
    near = best_factor(b[b.horizon <= 9].actual, b[b.horizon <= 9].f, c)
    far = best_factor(b[b.horizon >= 10].actual, b[b.horizon >= 10].f, c)
    cafe = best_factor(adf.actual, adf.f, c)
    hb = hold.f.sum() / hold.actual.sum() - 1
    paid = chosen["cost_plan"] <= chosen["cost_blend"]
    text = (f"<b>What the evidence says.</b> The tuned blend already leans slightly low: on the 2026 holdout it "
            f"came in {pct(abs(hb), 1, False)} {'under' if hb < 0 else 'over'} actual sales and above them in "
            f"{pct(chosen['over_blend'], 0, False)} of months. So the {c}:1 shade is modest, {p1(factor - 1)} "
            f"overall: {p1(near - 1)} for months 4–9 ahead and {p1(far - 1)} for months 10–15 ahead, where misses "
            f"grow. In 2026 it cut the months where the plan beat actual sales from "
            f"{pct(chosen['over_blend'], 0, False)} to {pct(chosen['over_plan'], 0, False)}"
            + (f" and lowered the {c}:1 cost ({pct(chosen['cost_blend'], 1, False)} → {pct(chosen['cost_plan'], 1, False)})."
               if paid else
               f"; because 2026 beat the forecast, it cost slightly more than it saved that year "
               f"({pct(chosen['cost_blend'], 1, False)} → {pct(chosen['cost_plan'], 1, False)}). It is insurance for "
               f"the year that disappoints.")
            + f" Single cafés miss by far more than the total, so a café target each café should beat three months "
              f"in four needs about {pct(cafe - 1, 0)}; such café targets add up to less than the brand plan by design.")
    return {"narrative": text}


# ------------------------------------------------------------------------------------------
# Tuning round
# ------------------------------------------------------------------------------------------
TUNING = OUTPUTS / "tuning"
SARIMA_TREND = {"drift": "Constant growth (drift)", "none": "No growth term", "local": "Local trend (re-estimated)"}


def _missing(v) -> bool:
    """None in a config, or the NaN pandas reads back for "None"/empty cells."""
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ("None", "nan", "NaN", "")


def hw_trend_label(row) -> str:
    if _missing(row["trend"]):
        return "No trend"
    if str(row["damped"]) == "False":
        return "Straight-line trend"
    return "Damped trend, φ estimated" if _missing(row["phi"]) else f"Damped trend, φ = {float(row['phi']):g}"


def describe(model: str, cfg: dict) -> list[tuple[str, str, str]]:
    """(setting, tuned value, round-1 value) in plain words."""
    base = DEFAULT_CONFIGS[model]
    on = lambda v: "on" if v else "off"  # noqa: E731
    if model == "Prophet":
        rows = [
            ("Trend flexibility", f"{cfg['changepoint_prior_scale']:.3g}", f"{base['changepoint_prior_scale']:.3g}"),
            ("Trend can bend until", f"{cfg['changepoint_range']:.0%} of history", f"{base['changepoint_range']:.0%} of history"),
            ("Seasonality strength", f"{cfg['seasonality_prior_scale']:.3g}", f"{base['seasonality_prior_scale']:.3g}"),
            ("Holiday strength", f"{cfg['holidays_prior_scale']:.3g}", f"{base['holidays_prior_scale']:.3g}"),
            ("Seasonality", cfg["seasonality_mode"], base["seasonality_mode"]),
            ("Yearly detail (Fourier terms)", str(cfg["yearly_fourier"]), str(base["yearly_fourier"])),
            ("Holiday periods", on(cfg["holiday_windows"]), on(base["holiday_windows"])),
            ("Summer weekly pattern", on(cfg["summer_weekly"]), on(base["summer_weekly"])),
            ("Cap catering spikes", on(cfg["cap_outliers"]), on(base["cap_outliers"])),
        ]
    elif model == "SARIMA":
        rows = [
            ("Growth term", SARIMA_TREND[cfg["trend"]], SARIMA_TREND[base["trend"]]),
            ("Orders searched", "12 (extended)" if cfg["candidates"] == "extended" else "6", "6"),
            ("Order chosen by", {"aicc": "AICc", "bic": "BIC"}[cfg["criterion"]],
             {"aicc": "AICc", "bic": "BIC"}[base["criterion"]]),
            ("Transform", cfg["transform"], base["transform"]),
            ("Trading days", cfg["td_adjust"], base["td_adjust"]),
        ]
    else:
        rows = [
            ("Trend", hw_trend_label(cfg), hw_trend_label(base)),
            ("Seasonality", {"mul": "multiplicative", "add": "additive"}[cfg["seasonal"]],
             {"mul": "multiplicative", "add": "additive"}[base["seasonal"]]),
            ("Transform", "log" if cfg["boxcox"] else "none", "log" if base["boxcox"] else "none"),
            ("Starting values", cfg["init"], base["init"]),
            ("Trading days", cfg["td_adjust"], base["td_adjust"]),
        ]
    return rows


def collect_tuning() -> dict:
    cfg = json.loads(TUNED_CONFIGS.read_text())
    hold = pd.read_csv(TUNING / "holdout_results.csv")
    blends = pd.read_csv(TUNING / "blend_results.csv")
    topdown = pd.read_csv(TUNING / "topdown_results.csv")
    by_setting = pd.read_csv(TUNING / "forecast_2027_by_setting.csv")
    trials = {"Prophet": pd.read_csv(TUNING / "prophet_trials.csv", dtype=str),
              "SARIMA": pd.read_csv(TUNING / "sarima_grid.csv", dtype=str),
              "Holt-Winters": pd.read_csv(TUNING / "hw_grid.csv", dtype=str)}
    for t in trials.values():
        t["objective"] = t["objective"].astype(float)

    t = {"design": cfg["design"], "adopted": cfg["adopted"], "best": cfg["best_on_tuning_set"],
         "adopted_models": [m for m in MODELS if cfg["adopted"]["variant"][m] == "tuned"]}
    rows = []
    for m in MODELS:
        h = hold[hold.model == m].set_index("variant")
        rows.append({"model": m, "adopted": cfg["adopted"]["variant"][m] == "tuned",
                     **{f"{v}_{k}": h.loc[v, f"holdout_{k}"] for v in ("baseline", "tuned")
                        for k in ("objective", "wape", "brand_wape", "brand_mape", "window_abs_error", "bias")},
                     **{f"tune_{v}": h.loc[v, "tune_objective"] for v in ("baseline", "tuned")}})
    t["models"] = rows
    b = blends.set_index(["models", "rule"])
    t["blend_rows"] = [{"models": mdl, "rule": rule, **b.loc[(mdl, rule)].to_dict()}
                       for mdl, rule in b.index]
    rule = cfg["adopted"]["blend"]
    t["blend_rule"] = rule if isinstance(rule, str) else "inverse-error weights"
    t["blend_weights"] = None if isinstance(rule, str) else {m: rule[m] / sum(rule.values()) for m in MODELS}
    t["blend_before"] = b.loc[("baseline models", "mean")].to_dict()
    t["blend_after"] = b.loc[("adopted models", t["blend_rule"])].to_dict()
    t["topdown"] = topdown.to_dict("records")
    t["by_setting"] = by_setting.to_dict("records")
    t["describe"] = {m: describe(m, {**DEFAULT_CONFIGS[m], **cfg["adopted"]["models"][m]}) for m in MODELS}
    t["describe_best"] = {m: describe(m, {**DEFAULT_CONFIGS[m], **cfg["best_on_tuning_set"][m]}) for m in MODELS}

    # one-at-a-time ablation from the best settings (src/feature_ablation.py)
    ab = pd.read_csv(TUNING / "feature_ablation.csv")
    feats = []
    for r in ab.itertuples():
        d_tune = r.tune_on / r.tune_off - 1
        d_hold = r.holdout_on / r.holdout_off - 1
        if abs(d_tune) < 0.01 and abs(d_hold) < 0.01:
            verdict = "no difference"
        elif d_tune < -0.01 and d_hold < -0.01:
            verdict = "helps"
        elif d_tune > 0.01 and d_hold > 0.01:
            verdict = "hurts"
        else:
            verdict = "mixed"
        feats.append({"model": r.model, "feature": r.feature, "in_best": bool(r.in_best),
                      "tune_on": r.tune_on, "tune_off": r.tune_off, "hold_on": r.holdout_on,
                      "hold_off": r.holdout_off, "d_tune": d_tune, "d_hold": d_hold, "verdict": verdict})
    t["features"] = feats
    t["sarima_trends"] = (trials["SARIMA"].groupby("trend").objective.min()
                          .rename(index=SARIMA_TREND).sort_values().to_dict())
    hw = trials["Holt-Winters"].assign(label=lambda d: d.apply(hw_trend_label, axis=1))
    t["hw_trends"] = hw.groupby("label").objective.min().sort_values().to_dict()
    t["n_prophet"] = len(trials["Prophet"])
    t["n_sarima"] = len(trials["SARIMA"])
    t["n_hw"] = len(trials["Holt-Winters"])
    from tune import HOLDOUT_ORIGINS, HOLDOUT_WINDOW, TUNE_ORIGINS, TUNE_WINDOW, Evaluator
    ev = Evaluator(load_stores(), 1)
    t["tune_store_months"] = sum(len(j[2]) for j in ev.jobs(TUNE_ORIGINS, TUNE_WINDOW))
    t["holdout_store_months"] = sum(len(j[2]) for j in ev.jobs(HOLDOUT_ORIGINS, HOLDOUT_WINDOW))
    td = pd.DataFrame(t["topdown"])
    t["topdown_rows"] = []
    for m in MODELS:
        g = td[td.model == m].set_index("approach")
        f = by_setting[(by_setting.model == m) & (by_setting.settings == "adopted")].set_index("approach")
        t["topdown_rows"].append({
            "model": m,
            "per_cafe_mape": g.loc["one model per cafe", "brand_mape"],
            "total_mape": g.loc["one model on the total", "brand_mape"],
            "per_cafe_miss": g.loc["one model per cafe", "window_abs_error"],
            "total_miss": g.loc["one model on the total", "window_abs_error"],
            "per_cafe_2027": f.loc["one model per cafe", "forecast_2027"],
            "total_2027": f.loc["one model on the total", "forecast_2027"],
        })
    base27 = by_setting[(by_setting.settings == "baseline")].set_index("model").forecast_2027
    t["baseline_2027"] = base27.to_dict()
    t["baseline_spread"] = float(base27.max() - base27.min())
    narrate(t)
    return t


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def narrate(t: dict) -> None:
    """Plain-English verdicts generated from the tuning numbers."""
    p1 = lambda x: pct(x, 1, False)  # noqa: E731
    adopted = [r for r in t["models"] if r["adopted"]]
    kept = [r for r in t["models"] if not r["adopted"]]
    parts = []
    if adopted:
        def gains(r):
            cafe = (r["baseline_wape"], r["tuned_wape"])
            brand = (r["baseline_brand_wape"], r["tuned_brand_wape"])
            better = [f"{lbl} error {p1(a)} → {p1(b)}" for lbl, (a, b) in (("café-month", cafe), ("brand", brand)) if b < a]
            worse = [f"{lbl} {p1(a)} → {p1(b)}" for lbl, (a, b) in (("café-month", cafe), ("brand", brand)) if b >= a]
            txt = f"{r['model']} improved its {_list(better)}"
            return txt + (f" while giving back a little at the other level ({_list(worse)})" if worse else "")
        parts.append("<b>Verdict.</b> On the 2026 holdout score, tuning won for " + _list([r["model"] for r in adopted])
                     + ". " + "; ".join(gains(r) for r in adopted) + ".")
    else:
        parts.append("<b>Verdict.</b> No tuned setting beat round 1 on the 2026 holdout.")
    if kept:
        parts.append(_list([r["model"] for r in kept]) + (" keeps" if len(kept) == 1 else " keep")
                     + " round-1 settings: the tuned version looked better on 2025 but did worse on 2026.")
    bb, ba = t["blend_before"], t["blend_after"]
    parts.append(f"<b>The blend</b> moves from {p1(bb['holdout_wape'])} to {p1(ba['holdout_wape'])} café-month error "
                 f"and from {p1(bb['holdout_brand_wape'])} to {p1(ba['holdout_brand_wape'])} at brand level; "
                 f"its average miss on the period total is {p1(ba['holdout_window_abs_error'])} "
                 f"(was {p1(bb['holdout_window_abs_error'])}).")
    t["verdict"] = " ".join(parts)

    f = pd.DataFrame(t["features"])
    helps = f[f.verdict == "helps"]
    hurts = f[f.verdict == "hurts"]
    lede = ("Starting from each model's best settings, every feature was switched on and off on its own and "
            "scored on 2025 and on the 2026 holdout. ")
    if len(helps):
        lede += "What helped: " + _list([f"{r.feature.split(':')[0].split(' (')[0].lower()} ({r.model})"
                                         for r in helps.itertuples()]) + ". "
    if len(hurts):
        lede += "What hurt: " + _list([f"{r.feature.split(':')[0].split(' (')[0].lower()} ({r.model})"
                                       for r in hurts.itertuples()]) + "."
    t["features_lede"] = lede.strip()

    rows = t["topdown_rows"]
    better_cafe = [r for r in rows if r["per_cafe_mape"] <= r["total_mape"]]
    gap27 = [r["total_2027"] / r["per_cafe_2027"] - 1 for r in rows]
    t["topdown_lede"] = ("Round 1 fitted every model once per café and added the results up. Here each model "
                         "is also fitted once to the cafés' combined sales, and both are scored on the same "
                         "2026 holdout.")
    verdict = (f"<b>One model per café wins for {len(better_cafe)} of 3 models</b> on brand monthly error"
               if len(better_cafe) >= 2 else
               f"<b>One model for all cafés wins for {3 - len(better_cafe)} of 3 models</b> on brand monthly error")
    verdict += (f". For 2027 the single models land within {pct(min(gap27), 1)} to {pct(max(gap27), 1)} of the "
                "per-café totals, so the headline barely depends on the choice. ")
    verdict += ("We keep one model per café: it is at least as accurate, it gives every café its own forecast, "
                "and it is not thrown off when a café opens (the combined series only starts in April 2024, "
                "when the eleventh café opened)." if len(better_cafe) >= 2 else
                "A single model is competitive on the total, but we keep one model per café: it gives every café "
                "its own forecast and is not thrown off when a café opens.")
    t["topdown_verdict"] = verdict


def date_span(a: str, b: str) -> str:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if a == b:
        return a.strftime("%b %-d, %Y")
    if a.month == b.month:
        return f"{a.strftime('%b %-d')}–{b.day}, {b.year}"
    return f"{a.strftime('%b %-d')} – {b.strftime('%b %-d, %Y')}"


# ------------------------------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------------------------------
def make_charts(n: dict) -> dict:
    out = {}
    rb = n["raw_brand"]
    hist = rb.loc[pd.Period("2022-10", "M"):pd.Period("2026-09", "M")]
    months = [p.to_timestamp() for p in hist.index]
    # (opening month, label, label offset in points)
    openings = [(pd.Timestamp("2023-07-01"), "Newport", (-4, 16, "right")),
                (pd.Timestamp("2023-09-01"), "Beverly Hills", (0, -20, "center")),
                (pd.Timestamp("2024-01-01"), "Brentwood", (8, -18, "left")),
                (pd.Timestamp("2024-03-01"), "Carlsbad", (-6, 14, "right"))]
    out["history"] = charts.history(months, hist.to_numpy(), openings)
    out["annual"] = charts.annual_columns(
        ["2023", "2024", "2025", "2026E", "2027F"],
        [n["a2023"], n["a2024"], n["a2025"], n["e2026"], n["f2027"][ENSEMBLE]],
        ["actual", "actual", "actual", "estimate", "forecast"])

    order = MODELS + [ENSEMBLE]
    bt = {r["key"]: r for r in n["bt"]}
    out["accuracy"] = charts.accuracy_panels(
        [LABEL[m] for m in order],
        [("Café-month error (WAPE)", [bt[m]["wape"] for m in order]),
         ("Brand monthly error (MAPE)", [bt[m]["mape"] for m in order]),
         ("Miss on the period total (avg.)", [bt[m]["abs_err"] for m in order])],
        best_label=BLEND)

    reh = []
    for origin, title in [("2024-09-30", "Rehearsal 1 · forecasting 2025 from Sep 2024"),
                          ("2025-09-30", "Rehearsal 5 · forecasting 2026 from Sep 2025")]:
        g = n["bt_df"][n["bt_df"].origin == pd.Timestamp(origin)].groupby("month")[
            ["actual"] + MODELS + [ENSEMBLE]].sum()
        mts = [pd.Period(m, "M").to_timestamp() for m in g.index]
        reh.append(charts.rehearsal(mts, g.actual.to_numpy(), g[ENSEMBLE].to_numpy(),
                                    g[MODELS].min(axis=1).to_numpy(), g[MODELS].max(axis=1).to_numpy(), title))
    out["rehearsal"] = reh

    act = rb.loc[pd.Period("2025-01", "M"):pd.Period("2026-09", "M")]
    fc = n["fc"]
    P = n["plan"]
    out["forecast"] = charts.forecast([p.to_timestamp() for p in act.index], act.to_numpy(),
                                      [p.to_timestamp() for p in fc.month], fc[ENSEMBLE].to_numpy(),
                                      fc.low.to_numpy(), fc.high.to_numpy(),
                                      plan=fc["Plan"].to_numpy() if P and "Plan" in fc else None)
    if P:
        plan_label = f"Plan (leans low, {P['over_cost']}:1)"
        out["dots"] = charts.model_dots(MODELS + ["Best estimate (blend)", plan_label],
                                        [n["f2027"][m] for m in order] + [P["f2027"]],
                                        n["e2026"], f"2026E {fm(n['e2026'])}", plan_label, size=(4.5, 3.0))
    else:
        out["dots"] = charts.model_dots([LABEL[m] for m in order], [n["f2027"][m] for m in order],
                                        n["e2026"], f"2026E {fm(n['e2026'])}", BLEND)
    g = sorted(n["cafes"], key=lambda c: -(c["f2027"] - c["e2026"]))
    out["growth"] = charts.growth_bars([c["store"] for c in g], [c["f2027"] - c["e2026"] for c in g],
                                       [c["growth"] for c in g])
    if P:
        plan = f"Plan (leans low, {P['over_cost']}:1)"
        scenarios = [
            ("Like-for-like cools to +8%", n["slow_case"]),
            (f"Conservative: {n['low_model']}", n["low"]),
            (plan, P["f2027"]),
            ("Best estimate: blend", n["f2027"][ENSEMBLE]),
            (f"Stretch: {n['high_model']}", n["high"]),
            ("Plan + 1 new café from July*", P["f2027"] + n["new_cafe_half_year"]),
        ]
    else:
        plan = "Plan: blend of 3 models"
        scenarios = [
            ("Like-for-like cools to +8%", n["slow_case"]),
            (f"Conservative: {n['low_model']}", n["low"]),
            (plan, n["f2027"][ENSEMBLE]),
            (f"Stretch: {n['high_model']}", n["high"]),
            ("Plan + 1 new café from July*", n["f2027"][ENSEMBLE] + n["new_cafe_half_year"]),
        ]
    out["scenarios"] = charts.model_dots([s for s, _ in scenarios], [v for _, v in scenarios],
                                         n["e2026"], f"2026E {fm(n['e2026'])}", plan, size=(5.6, 2.9),
                                         name="scenarios")
    if n["tuning"]:
        tdr = n["tuning"]["topdown_rows"]
        out["topdown"] = charts.grouped_hbars(
            MODELS, [("One model per café", [r["per_cafe_mape"] for r in tdr], charts.INK),
                     ("One model on the total", [r["total_mape"] for r in tdr], charts.G400)],
            size=(4.6, 2.3), name="topdown")
    if P:
        adf = pd.read_csv(TUNING / "adopted_forecasts.csv")
        h = adf[(adf.set == "holdout") & (adf.origin == "2025-09-30")].groupby("month")[["actual", "f"]].sum()
        k25 = P["chosen"]["factor_tuning"]
        out["plan_backtest"] = charts.plan_backtest(
            [pd.Period(m, "M").to_timestamp() for m in h.index], h.actual.to_numpy(), h.f.to_numpy(),
            h.f.to_numpy() * k25, "Forecast made Sep 30, 2025 for Jan–Sep 2026 (9 cafés)")
        rows = P["rows"]
        labels = ["Best estimate (blend)"] + [f"{r['over_cost']}:1" + (" · plan" if r["chosen"] else "") for r in rows]
        values = [n["f2027"][ENSEMBLE]] + [r["f2027"] for r in rows]
        chosen_label = next(lbl for lbl in labels if lbl.endswith("· plan"))
        out["plan_ladder"] = charts.model_dots(labels, values, n["e2026"], f"2026E {fm(n['e2026'])}",
                                               chosen_label, size=(4.4, 2.9), name="plan-ladder")
    if n["tuning"]:
        T = n["tuning"]
        rows, bb, ba = T["models"], T["blend_before"], T["blend_after"]
        out["dumbbells"] = charts.dumbbells(
            MODELS + ["Blend"],
            [("Café-month error (WAPE) on the 2026 holdout",
              [r["baseline_wape"] for r in rows] + [bb["holdout_wape"]],
              [r["tuned_wape"] for r in rows] + [ba["holdout_wape"]]),
             ("Brand monthly error (WAPE) on the 2026 holdout",
              [r["baseline_brand_wape"] for r in rows] + [bb["holdout_brand_wape"]],
              [r["tuned_brand_wape"] for r in rows] + [ba["holdout_brand_wape"]])])
    return out


# ------------------------------------------------------------------------------------------
def render(n: dict, svgs: dict) -> Path:
    env = Environment(loader=FileSystemLoader(ROOT / "report"), autoescape=False)
    env.filters.update(fm=fm, fk=fk, pct=pct)
    html = env.get_template("template.html.j2").render(n=n, svg=svgs, MODELS=MODELS, ENSEMBLE=ENSEMBLE,
                                                        LABEL=LABEL, BLEND=BLEND)
    path = BUILD / "parakeet_2027_sales_forecast.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", help="path to the parakeet-cafe-design skill folder")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    ds = find_design_system(args.ds)
    charts.setup(copy_assets(ds))
    n = collect()
    html = render(n, make_charts(n))
    print(f"HTML -> {html.relative_to(ROOT)}")
    if not args.no_pdf:
        # Playwright is resolved from the global node_modules
        node_path = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
        env = {**os.environ, "NODE_PATH": node_path}
        subprocess.run(["node", str(ROOT / "report" / "render_pdf.cjs"), str(html), str(PDF)],
                       check=True, env=env)
        print(f"PDF  -> {PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
