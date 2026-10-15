"""One-feature-at-a-time ablation from each model's best tuned settings.

The tuning search answers "which combination is best"; this answers "what does each feature
do on its own", fairly: starting from the best settings found by src/tune.py, each feature is
switched on and off with everything else fixed, and both versions are scored on the 2025
tuning set and on the Jan-Sep 2026 holdout.

    python src/feature_ablation.py      (after src/tune.py)
Writes outputs/tuning/feature_ablation.csv.
"""
from __future__ import annotations

import json

import pandas as pd

from forecasting import DEFAULT_CONFIGS, TUNED_CONFIGS, load_stores
from tune import (HOLDOUT_ORIGINS, HOLDOUT_WINDOW, OUT, TUNE_ORIGINS, TUNE_WINDOW, Evaluator,
                  scores)

# model -> (feature key, label, value on, value off)
FEATURES = {
    "Prophet": [
        ("holiday_windows", "Holiday periods: Christmas week, spring break, Thanksgiving weekend", True, False),
        ("summer_weekly", "Separate weekly pattern for summer", True, False),
        ("cap_outliers", "Cap catering spikes at 3× the 29-day median", 3.0, 0.0),
        ("seasonality_mode", "Multiplicative rather than additive seasonality", "multiplicative", "additive"),
    ],
    "SARIMA": [
        ("td_adjust", "Weekday-weighted trading days (vs plain day count)", "weekday", "calendar"),
        ("transform", "Log transform", "log", "none"),
    ],
    "Holt-Winters": [
        ("td_adjust", "Weekday-weighted trading days (vs plain day count)", "weekday", "calendar"),
        ("boxcox", "Log transform", "log", False),
        ("seasonal", "Multiplicative rather than additive seasonality", "mul", "add"),
    ],
}


def main() -> None:
    best = json.loads(TUNED_CONFIGS.read_text())["best_on_tuning_set"]
    ev = Evaluator(load_stores(), 4)
    rows = []
    for model, feats in FEATURES.items():
        base = {**DEFAULT_CONFIGS[model], **best[model]}
        for key, label, on, off in feats:
            res = {}
            for state, value in (("on", on), ("off", off)):
                cfg = {**base, key: value}
                res[f"tune_{state}"] = scores(ev.run(model, cfg, TUNE_ORIGINS, TUNE_WINDOW))["objective"]
                res[f"holdout_{state}"] = scores(ev.run(model, cfg, HOLDOUT_ORIGINS, HOLDOUT_WINDOW))["objective"]
            rows.append({"model": model, "feature": label, "key": key,
                         "in_best": base[key] == on or (key == "cap_outliers" and bool(base[key])),
                         **res})
            print(f"{model:13s} {key:17s} tune {res['tune_on']:.4f}/{res['tune_off']:.4f} "
                  f"holdout {res['holdout_on']:.4f}/{res['holdout_off']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "feature_ablation.csv", index=False)


if __name__ == "__main__":
    main()
