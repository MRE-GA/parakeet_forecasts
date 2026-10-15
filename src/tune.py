"""Hyperparameter and feature tuning, validated on a holdout the search never sees.

Tuning set : forecasts made at month-ends Sep 2024, Dec 2024, Mar 2025 and Jun 2025, scored on
             months 4-15 ahead that fall in 2025 or earlier.
Holdout    : forecasts made at Jun 2025, Sep 2025 and Dec 2025, scored only on Jan-Sep 2026.

Steps
 1. Prophet: Optuna TPE search over hyperparameters and feature switches (baseline tried first).
 2. SARIMA and Holt-Winters: full grid over their options.
 3. Holdout: baseline vs best settings per model. Tuned settings are adopted only if they lower
    the holdout objective (mean of cafe-month and brand-month WAPE).
 4. Blend rule (mean / median / inverse-error weights) picked on the tuning set and kept only if
    it is at least as good as the plain mean on the holdout.
 5. Top-down check: one model fitted to the cafes' total vs one model per cafe (bottom-up).
 6. Conservative plan: over-forecasting sales costs more than under-forecasting, so the plan is
    the blend times a factor k that minimises an asymmetric cost (over-forecast dollars weighted
    OVER_COST x). k is fitted on the tuning set's brand-month errors and checked on the holdout;
    the 2027 plan uses k re-estimated on both. Model settings stay chosen on accuracy alone, so
    nothing is deliberately underfit.

Writes outputs/tuning/*.csv and outputs/tuned_configs.json (read by forecast_2027.py).

    python src/tune.py [--trials 150] [--jobs 4] [--reuse prophet,sarima]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from backtest import eligible
from forecasting import (DEFAULT_CONFIGS, FORECASTERS, MODELS, OUTPUTS, STORES, TUNED_CONFIGS,
                         aggregate_store, bias, combine, load_stores, mape, wape)

TUNE_ORIGINS = ["2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"]
HOLDOUT_ORIGINS = ["2025-06-30", "2025-09-30", "2025-12-31"]
TUNE_WINDOW = (None, pd.Period("2025-12", "M"))
HOLDOUT_WINDOW = (pd.Period("2026-01", "M"), pd.Period("2026-09", "M"))
FINAL_ORIGIN = pd.Timestamp("2026-09-30")
MIN_H, MAX_H = 4, 15
OUT = OUTPUTS / "tuning"
OVER_COST = 3  # an over-forecast dollar costs this many under-forecast dollars (default plan)
COST_RATIOS = [1, 2, 3, 4]


def asym_cost(actual, forecast, over_cost: float) -> float:
    """Asymmetric absolute error as a share of actual sales: over-forecasts weighted over_cost x."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    err = forecast - actual
    return (over_cost * err.clip(min=0) - err.clip(max=0)).sum() / actual.sum()


def best_factor(actual, forecast, over_cost: float) -> float:
    """Multiplier on the forecast that minimises the asymmetric cost (grid, 0.1% steps)."""
    grid = np.arange(0.80, 1.101, 0.001)
    costs = [asym_cost(actual, forecast * k, over_cost) for k in grid]
    return float(grid[int(np.argmin(costs))])


def calibrate_plan(tune_brand: pd.DataFrame, hold_brand: pd.DataFrame) -> pd.DataFrame:
    """Plan factor per cost ratio: fitted on tuning brand-months, scored on the holdout."""
    rows = []
    both = pd.concat([tune_brand, hold_brand])
    for c in COST_RATIOS:
        k = best_factor(tune_brand.actual, tune_brand.f, c)
        h_a, h_f = hold_brand.actual.to_numpy(), hold_brand.f.to_numpy()
        rows.append({
            "over_cost": c, "factor_tuning": k,
            "factor_final": best_factor(both.actual, both.f, c),
            "holdout_cost_blend": asym_cost(h_a, h_f, c), "holdout_cost_plan": asym_cost(h_a, h_f * k, c),
            "holdout_over_share_blend": float(np.mean(h_f > h_a)),
            "holdout_over_share_plan": float(np.mean(h_f * k > h_a)),
            "holdout_bias_plan": bias(h_a, h_f * k),
        })
    return pd.DataFrame(rows)

SARIMA_GRID = [
    {"trend": t, "candidates": cs, "criterion": cr, "transform": tr, "td_adjust": td}
    for t, cs, cr, tr, td in itertools.product(
        ["drift", "none", "local"], ["small", "extended"], ["aicc", "bic"], ["log", "none"],
        ["weekday", "calendar"])
]
_HW_TRENDS = [
    {"trend": None, "damped": False, "phi": None},
    {"trend": "add", "damped": False, "phi": None},
    {"trend": "add", "damped": True, "phi": None},
    {"trend": "add", "damped": True, "phi": 0.9},
    {"trend": "add", "damped": True, "phi": 0.95},
    {"trend": "add", "damped": True, "phi": 0.98},
]
HW_GRID = [
    {**t, "seasonal": s, "boxcox": b, "init": i, "td_adjust": td}
    for t, s, b, i, td in itertools.product(
        _HW_TRENDS, ["mul", "add"], [False, "log"], ["estimated", "heuristic"], ["weekday", "calendar"])
]


def prophet_space(trial: optuna.Trial) -> dict:
    return {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
        "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["multiplicative", "additive"]),
        "changepoint_range": trial.suggest_float("changepoint_range", 0.7, 0.95),
        "yearly_fourier": trial.suggest_categorical("yearly_fourier", [4, 6, 8, 10, 15, 20]),
        "holiday_windows": trial.suggest_categorical("holiday_windows", [False, True]),
        "summer_weekly": trial.suggest_categorical("summer_weekly", [False, True]),
        "cap_outliers": trial.suggest_categorical("cap_outliers", [0.0, 3.0]),  # 0 = off
    }


PROPHET_BASELINE = {k: DEFAULT_CONFIGS["Prophet"][k] for k in [
    "changepoint_prior_scale", "seasonality_prior_scale", "holidays_prior_scale", "seasonality_mode",
    "changepoint_range", "yearly_fourier", "holiday_windows", "summer_weekly"]} | {"cap_outliers": 0.0}


# --------------------------------------------------------------------------------------------
def target_months(origin: pd.Timestamp, window) -> list[pd.Period]:
    lo, hi = window
    months = [origin.to_period("M") + h for h in range(MIN_H, MAX_H + 1)]
    return [m for m in months if (lo is None or m >= lo) and m <= hi]


def _fit(model, cfg, sd, origin, months):
    try:
        point, _ = FORECASTERS[model](sd, origin, max(months), cfg=cfg)
        return point.reindex(months).to_numpy()
    except Exception:
        return np.full(len(months), np.nan)


class Evaluator:
    def __init__(self, stores, n_jobs):
        self.stores = stores
        self.n_jobs = n_jobs
        self.actual = {s: sd.monthly_actual() for s, sd in stores.items()}

    def jobs(self, origins, window):
        out = []
        for o in origins:
            o = pd.Timestamp(o)
            months = target_months(o, window)
            out += [(sd, o, months) for sd in self.stores.values() if eligible(sd, o) and months]
        return out

    def run(self, model, cfg, origins, window) -> pd.DataFrame:
        jobs = self.jobs(origins, window)
        preds = Parallel(n_jobs=self.n_jobs)(delayed(_fit)(model, cfg, sd, o, m) for sd, o, m in jobs)
        rows = []
        for (sd, o, months), p in zip(jobs, preds):
            for month, f in zip(months, p):
                rows.append((o, sd.store, month, (month - o.to_period("M")).n,
                             self.actual[sd.store].get(month, np.nan), f))
        return pd.DataFrame(rows, columns=["origin", "store", "month", "horizon", "actual", "forecast"])


def scores(df: pd.DataFrame, col: str = "forecast") -> dict:
    """Objective = mean of cafe-month WAPE and brand-month WAPE, so a setting can't win by
    sharpening cafes while drifting on the total (or the reverse)."""
    if df[col].isna().any():
        return {k: np.nan for k in ("objective", "wape", "brand_wape", "brand_mape", "window_abs_error", "bias")}
    brand = df.groupby(["origin", "month"])[["actual", col]].sum().reset_index()
    window = [bias(g.actual, g[col]) for _, g in brand.groupby("origin")]
    store_w, brand_w = wape(df.actual, df[col]), wape(brand.actual, brand[col])
    return {"objective": (store_w + brand_w) / 2, "wape": store_w, "brand_wape": brand_w,
            "brand_mape": mape(brand.actual, brand[col]),
            "window_abs_error": float(np.mean(np.abs(window))), "bias": bias(df.actual, df[col])}


def grid_search(ev: Evaluator, model: str, grid: list[dict]) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(grid):
        t = time.time()
        s = scores(ev.run(model, cfg, TUNE_ORIGINS, TUNE_WINDOW))
        rows.append({**{k: str(v) for k, v in cfg.items()}, **s, "config": json.dumps(cfg)})
        print(f"  {model} {i + 1}/{len(grid)} objective={s['objective']:.4f} ({time.time() - t:.0f}s)", flush=True)
    return pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)


def prophet_search(ev: Evaluator, n_trials: int) -> pd.DataFrame:
    def objective(trial):
        s = scores(ev.run("Prophet", prophet_space(trial), TUNE_ORIGINS, TUNE_WINDOW))
        for k, v in s.items():
            trial.set_user_attr(k, v)
        return s["objective"] if np.isfinite(s["objective"]) else 1.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=20))
    study.enqueue_trial(PROPHET_BASELINE)
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[lambda st, tr: print(f"  Prophet trial {tr.number + 1}/{n_trials} "
                                                   f"objective={tr.value:.4f} best={st.best_value:.4f}", flush=True)])
    rows = [{**tr.params, **tr.user_attrs, "trial": tr.number, "config": json.dumps(tr.params)}
            for tr in study.trials]
    return pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)


def main(n_trials: int, n_jobs: int, reuse: set[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stores = load_stores()
    ev = Evaluator(stores, n_jobs)
    t0 = time.time()

    # 1-2. search on the tuning set (or reuse a finished search) ---------------------------------
    if "prophet" in reuse:
        prophet = pd.read_csv(OUT / "prophet_trials.csv")
        n_trials = len(prophet)
    else:
        print("Prophet search", flush=True)
        prophet = prophet_search(ev, n_trials)
        prophet.to_csv(OUT / "prophet_trials.csv", index=False)
    if "sarima" in reuse:
        sarima = pd.read_csv(OUT / "sarima_grid.csv")
    else:
        print("SARIMA grid", flush=True)
        sarima = grid_search(ev, "SARIMA", SARIMA_GRID)
        sarima.to_csv(OUT / "sarima_grid.csv", index=False)
    if "hw" in reuse:
        hw = pd.read_csv(OUT / "hw_grid.csv")
    else:
        print("Holt-Winters grid", flush=True)
        hw = grid_search(ev, "Holt-Winters", HW_GRID)
        hw.to_csv(OUT / "hw_grid.csv", index=False)

    best = {"Prophet": json.loads(prophet.config[0]), "SARIMA": json.loads(sarima.config[0]),
            "Holt-Winters": json.loads(hw.config[0])}
    baseline = {m: dict(DEFAULT_CONFIGS[m]) for m in MODELS}

    # 3. holdout: baseline vs tuned per model --------------------------------------------------
    hold, tune_fc, hold_fc = [], {}, {}
    for m in MODELS:
        for variant, cfg in (("baseline", baseline[m]), ("tuned", best[m])):
            tune_fc[(m, variant)] = ev.run(m, cfg, TUNE_ORIGINS, TUNE_WINDOW)
            hold_fc[(m, variant)] = ev.run(m, cfg, HOLDOUT_ORIGINS, HOLDOUT_WINDOW)
            hold.append({"model": m, "variant": variant,
                         **{f"tune_{k}": v for k, v in scores(tune_fc[(m, variant)]).items()},
                         **{f"holdout_{k}": v for k, v in scores(hold_fc[(m, variant)]).items()},
                         "config": json.dumps(cfg)})
    hold = pd.DataFrame(hold)
    adopted_variant = {}
    for m in MODELS:
        h = hold[hold.model == m].set_index("variant")
        adopted_variant[m] = "tuned" if h.loc["tuned", "holdout_objective"] < h.loc["baseline", "holdout_objective"] else "baseline"
    adopted = {m: best[m] if adopted_variant[m] == "tuned" else baseline[m] for m in MODELS}

    # 4. blend rule ------------------------------------------------------------------------------
    def frame(fc: dict, variants: dict) -> pd.DataFrame:
        base = fc[(MODELS[0], variants[MODELS[0]])][["origin", "store", "month", "horizon", "actual"]].copy()
        for m in MODELS:
            base[m] = fc[(m, variants[m])].forecast.to_numpy()
        return base

    blends = []
    for label, variants in (("baseline models", {m: "baseline" for m in MODELS}),
                            ("adopted models", adopted_variant)):
        tf, hf = frame(tune_fc, variants), frame(hold_fc, variants)
        inv = {m: 1 / scores(tf, m)["objective"] for m in MODELS}
        for rule_name, rule in (("mean", "mean"), ("median", "median"), ("inverse-error weights", inv)):
            blends.append({"models": label, "rule": rule_name, "rule_value": json.dumps(rule),
                           **{f"tune_{k}": v for k, v in scores(tf.assign(f=combine(tf, rule)), "f").items()},
                           **{f"holdout_{k}": v for k, v in scores(hf.assign(f=combine(hf, rule)), "f").items()}})
    blends = pd.DataFrame(blends)
    ad = blends[blends.models == "adopted models"].set_index("rule")
    pick = ad.tune_objective.idxmin()
    blend_rule = json.loads(ad.loc[pick, "rule_value"]) if ad.loc[pick, "holdout_objective"] <= ad.loc["mean", "holdout_objective"] else "mean"

    # 6. conservative plan factor on the adopted blend ------------------------------------------
    tf, hf = frame(tune_fc, adopted_variant), frame(hold_fc, adopted_variant)
    tf["f"], hf["f"] = combine(tf, blend_rule), combine(hf, blend_rule)
    pd.concat([tf.assign(set="tuning"), hf.assign(set="holdout")]).to_csv(
        OUT / "adopted_forecasts.csv", index=False)
    brand = lambda d: d.groupby(["origin", "month"])[["actual", "f"]].sum().reset_index()  # noqa: E731
    plan = calibrate_plan(brand(tf), brand(hf))
    plan.to_csv(OUT / "plan_calibration.csv", index=False)
    plan_factor = float(plan.set_index("over_cost").loc[OVER_COST, "factor_final"])

    # 5. top-down check on the holdout -----------------------------------------------------------
    td_rows = []
    for o in HOLDOUT_ORIGINS:
        o = pd.Timestamp(o)
        group = [sd for sd in stores.values() if eligible(sd, o)]
        total = aggregate_store(group)
        months = target_months(o, HOLDOUT_WINDOW)
        actual = sum(ev.actual[sd.store] for sd in group).reindex(months).to_numpy()
        for m in MODELS:
            p, _ = FORECASTERS[m](total, o, max(months), cfg=adopted[m])
            bottom_up = hold_fc[(m, adopted_variant[m])]
            bu = bottom_up[bottom_up.origin == o].groupby("month").forecast.sum().reindex(months).to_numpy()
            for month, a, f_td, f_bu in zip(months, actual, p.reindex(months).to_numpy(), bu):
                td_rows.append({"origin": o, "month": month, "model": m, "actual": a,
                                "one model on the total": f_td, "one model per cafe": f_bu})
    td = pd.DataFrame(td_rows)
    td_summary = []
    for m in MODELS:
        g = td[td.model == m]
        for approach in ("one model per cafe", "one model on the total"):
            win = [bias(x.actual, x[approach]) for _, x in g.groupby("origin")]
            td_summary.append({"model": m, "approach": approach, "brand_mape": mape(g.actual, g[approach]),
                               "window_abs_error": float(np.mean(np.abs(win)))})
    td_summary = pd.DataFrame(td_summary)

    # 2027 totals under each setting (final fit on data through Sep 2026) ----------------------
    final = []
    months27 = pd.period_range("2027-01", "2027-12", freq="M")
    all_total = aggregate_store(list(stores.values()))
    for m in MODELS:
        for variant, cfg in (("baseline", baseline[m]), ("adopted", adopted[m])):
            pts = Parallel(n_jobs=n_jobs)(delayed(_fit)(m, cfg, sd, FINAL_ORIGIN, list(months27))
                                          for sd in stores.values())
            final.append({"model": m, "settings": variant, "approach": "one model per cafe",
                          "forecast_2027": float(np.sum(pts))})
        p, _ = FORECASTERS[m](all_total, FINAL_ORIGIN, months27[-1], cfg=adopted[m])
        final.append({"model": m, "settings": "adopted", "approach": "one model on the total",
                      "forecast_2027": float(p.reindex(months27).sum())})
    final = pd.DataFrame(final)

    hold.to_csv(OUT / "holdout_results.csv", index=False)
    blends.to_csv(OUT / "blend_results.csv", index=False)
    td.to_csv(OUT / "topdown_holdout_forecasts.csv", index=False)
    td_summary.to_csv(OUT / "topdown_results.csv", index=False)
    final.to_csv(OUT / "forecast_2027_by_setting.csv", index=False)
    TUNED_CONFIGS.write_text(json.dumps({
        "design": {"tune_origins": TUNE_ORIGINS, "tune_targets_through": str(TUNE_WINDOW[1]),
                   "holdout_origins": HOLDOUT_ORIGINS,
                   "holdout_targets": [str(HOLDOUT_WINDOW[0]), str(HOLDOUT_WINDOW[1])],
                   "horizons": [MIN_H, MAX_H], "prophet_trials": n_trials,
                   "sarima_grid": len(SARIMA_GRID), "hw_grid": len(HW_GRID)},
        "best_on_tuning_set": best,
        "adopted": {"models": {m: adopted[m] for m in MODELS}, "blend": blend_rule,
                    "variant": adopted_variant},
        "plan": {"over_cost": OVER_COST, "factor": plan_factor},
    }, indent=2, default=str))

    pd.set_option("display.width", 220)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(hold.drop(columns="config").round(4).to_string(index=False))
    print(blends.drop(columns="rule_value").round(4).to_string(index=False))
    print(td_summary.round(4).to_string(index=False))
    print(final.round(0).to_string(index=False))
    print(plan.round(4).to_string(index=False))
    print("adopted:", json.dumps({"variant": adopted_variant, "blend": blend_rule,
                                  "plan_factor": plan_factor}, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reuse", default="", help="comma list of finished searches to reload: prophet,sarima,hw")
    a = ap.parse_args()
    main(a.trials, a.jobs, {x.strip() for x in a.reuse.split(",") if x.strip()})
