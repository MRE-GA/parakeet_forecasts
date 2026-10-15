"""Build the Parakeet-branded PDF report from the forecast outputs.

    python report/build_report.py [--ds PATH_TO_PARAKEET_DESIGN_SYSTEM]

Run src/backtest.py, src/prophet_sensitivity.py and src/forecast_2027.py first.
Brand fonts, logo and photography come from the Parakeet Cafe design-system skill (not
committed to this repo); pass its folder with --ds or the PARAKEET_DS environment variable.
The PDF is printed by headless Chromium through Playwright (report/render_pdf.mjs).
"""
from __future__ import annotations

import argparse
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
from forecasting import ANOMALIES, MODELS, OUTPUTS, REGION, STORE_START, STORES, load_daily  # noqa: E402

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
    sens = pd.read_csv(OUTPUTS / "prophet_sensitivity.csv")
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

    lv = live[["actual_oct_1_14"] + all_models].sum()
    n["live"] = {"actual": lv.actual_oct_1_14, **{m: lv[m] / lv.actual_oct_1_14 - 1 for m in all_models}}

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
        })
    cafes.sort(key=lambda c: -c["f2027"])
    n["cafes"] = cafes
    top = sorted(cafes, key=lambda c: -(c["f2027"] - c["e2026"]))
    n["growth_total"] = n["f2027"][ENSEMBLE] - n["e2026"]
    n["top3_share"] = sum(c["f2027"] - c["e2026"] for c in top[:3]) / n["growth_total"]
    n["top3"] = [c["store"] for c in top[:3]]

    # brand months
    fc = brand_fc.assign(month=pd.PeriodIndex(brand_fc.month, freq="M"))
    fc["low"] = fc[MODELS].min(axis=1)
    fc["high"] = fc[MODELS].max(axis=1)
    raw_brand = raw.groupby(raw.date.dt.to_period("M")).net_sales.sum()
    # 2026 reference for each 2027 month: actual Jan-Sep, blended estimate Oct-Dec
    ref26 = {p: raw_brand[p] if p.month <= 9 else fc.set_index("month").loc[p, ENSEMBLE]
             for p in pd.period_range("2026-01", "2026-12", freq="M")}
    n["months27"] = [{"month": r.month.strftime("%b"), **{m: r[m] for m in all_models},
                      "ref2026": ref26[r.month - 12]}
                     for _, r in fc[fc.month.dt.year == 2027].iterrows()]
    growth = [r[ENSEMBLE] / r["ref2026"] - 1 for r in n["months27"]]
    n["m_growth_min"], n["m_growth_max"] = min(growth), max(growth)
    q1 = n["months27"][:3]
    n["q1_growth"] = sum(r[ENSEMBLE] for r in q1) / sum(r["ref2026"] for r in q1) - 1
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
    n["sens"] = sens
    n["bt_df"] = bt
    n["anomalies"] = [f"{who or 'All cafés'}: {date_span(a, b)} ({why})" for who, a, b, why in ANOMALIES]
    n["fold_label"] = {o: pd.Timestamp(o).strftime("%b %-d, %Y") for o in n["bt_folds"].index}
    return n


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
    out["forecast"] = charts.forecast([p.to_timestamp() for p in act.index], act.to_numpy(),
                                      [p.to_timestamp() for p in fc.month], fc[ENSEMBLE].to_numpy(),
                                      fc.low.to_numpy(), fc.high.to_numpy())
    out["dots"] = charts.model_dots([LABEL[m] for m in order], [n["f2027"][m] for m in order],
                                    n["e2026"], f"2026E {fm(n['e2026'])}", BLEND)
    g = sorted(n["cafes"], key=lambda c: -(c["f2027"] - c["e2026"]))
    out["growth"] = charts.growth_bars([c["store"] for c in g], [c["f2027"] - c["e2026"] for c in g],
                                       [c["growth"] for c in g])
    plan = f"Plan: blend of 3 models"
    scenarios = [
        ("Like-for-like cools to +8%", n["slow_case"]),
        ("Conservative: Holt-Winters", n["f2027"]["Holt-Winters"]),
        (plan, n["f2027"][ENSEMBLE]),
        ("Stretch: SARIMA", n["f2027"]["SARIMA"]),
        ("Plan + 1 new café from July*", n["f2027"][ENSEMBLE] + n["new_cafe_half_year"]),
    ]
    out["scenarios"] = charts.model_dots([s for s, _ in scenarios], [v for _, v in scenarios],
                                         n["e2026"], f"2026E {fm(n['e2026'])}", plan, size=(5.6, 2.55),
                                         name="scenarios")
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
