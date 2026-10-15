"""Build the 2027 budget-review deck: a non-technical, Parakeet-branded PDF that walks through the
investor budget ("2027 Sales Objectives by Café – Investors", October 2026) against Toast sales
and our 2027 forecast: the 2026 finish, the 2027 objectives, then two pages per café.

    python report/build_deck.py [--ds PATH_TO_PARAKEET_DESIGN_SYSTEM] [--no-pdf]

Inputs: data/cfo_budget_2027.csv and data/cfo_budget_2027_comp_trend.csv (the budget, keyed in
from the PDF), data/raw/parakeet_toast_daily.csv and the forecast outputs (run
src/forecast_2027.py first). Outputs: outputs/Parakeet_2027_Budget_Review.pdf and
outputs/budget_review_by_cafe.csv.
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
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report"))

import charts  # noqa: E402
import deck_charts as dc  # noqa: E402
from build_report import BUILD, copy_assets, find_design_system  # noqa: E402
from forecasting import MODELS, OUTPUTS, STORE_START, load_configs, load_daily, load_stores  # noqa: E402

DATA = ROOT / "data"
PDF = OUTPUTS / "Parakeet_2027_Budget_Review.pdf"
CSV = OUTPUTS / "budget_review_by_cafe.csv"
PHOTOS = ["photos/hero-3.jpg", "photos/split-order.jpg", "photos/hero-1.jpg", "photos/tout-mission.jpg"]
BLEND, PLAN = "Average of 3", "Plan"
ORIGIN = pd.Timestamp("2026-09-30")  # last full month the forecast was fitted on
LIVE_END = pd.Timestamp("2026-10-14")  # last day of Toast data
ORIGINAL = ["Carmel Valley", "Coronado", "La Jolla", "La Jolla Juicery", "Little Italy"]
ANALOGS = ["Newport", "Beverly Hills", "Brentwood", "Carlsbad"]  # our last four openings

# Our read of each café, set after reviewing the evidence on its two pages (check_verdicts()
# re-tests the numbers behind each call so a data refresh cannot silently contradict the text).
VERDICT = {
    "Beverly Hills": "too_low", "Brentwood": "defensible", "Carlsbad": "too_low", "Coronado": "defensible",
    "Carmel Valley": "too_low", "La Jolla": "slightly_low", "La Jolla Juicery": "in_line",
    "Little Italy": "too_low", "Newport": "too_low", "La Jolla Cove": "above", "Nautilus": "missing",
}
VERDICT_LABEL = {"too_low": "Objective too low", "slightly_low": "Slightly low", "defensible": "Defensible",
                 "in_line": "In line", "above": "Budget above our view", "missing": "Not in the budget"}
SHORT = {"Carmel Valley": "Del Mar Café", "La Jolla": "La Jolla Café", "Little Italy": "Little Italy",
         "Newport": "Newport Beach"}


# ------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------
def fm(x: float, d: int = 2) -> str:
    return f"${x / 1e6:,.{d}f}M"


def fk(x: float) -> str:
    return f"${abs(x) / 1e3:,.0f}K"


def sfk(x: float) -> str:
    """Signed thousands: +$362K / −$5K."""
    return f"{'+' if x >= 0 else '−'}${abs(x) / 1e3:,.0f}K"


def sfm(x: float, d: int = 2) -> str:
    return f"{'+' if x >= 0 else '−'}${abs(x) / 1e6:,.{d}f}M"


def words(k: int) -> str:
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve"][k] if 0 <= k <= 12 else str(k)


def frange(lo: float, hi: float) -> str:
    return f"${lo / 1e6:.2f}–{hi / 1e6:.2f}M"


def pct(x: float, d: int = 1) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.{d}f}%"


def pct0(x: float) -> str:
    return pct(x, 0)


def apct(x: float, d: int = 1) -> str:
    """Unsigned percent, for phrases like "up 14%"."""
    return f"{abs(x) * 100:.{d}f}%"


def qlabel(p: pd.Period) -> str:
    return f"Q{p.quarter} {p.year}"


# ------------------------------------------------------------------------------------------
# Evidence
# ------------------------------------------------------------------------------------------
def window(raw: pd.DataFrame, store: str | list, a: str, z: str) -> tuple[float, float]:
    stores = [store] if isinstance(store, str) else store
    d = raw[raw.store.isin(stores) & (raw.date >= a) & (raw.date <= z)]
    return d.net_sales.sum(), d.checks.sum()


def quarterly_yoy(raw: pd.DataFrame, store: str) -> pd.Series:
    """Net sales growth vs the same quarter a year earlier, for quarters the café traded fully."""
    d = raw[(raw.store == store) & (raw.date <= ORIGIN)]
    q = d.groupby(d.date.dt.to_period("Q")).net_sales.sum()
    q = q.reindex(pd.period_range(q.index.min(), q.index.max(), freq="Q"))
    start = pd.Timestamp(STORE_START[store])
    first_full = start.to_period("Q") + (0 if start == start.to_period("Q").start_time else 1)
    yoy = (q / q.shift(4) - 1).loc[max(first_full + 4, pd.Period("2023Q4", "Q")):]
    return yoy.dropna()


def hold_path(sd, origin: pd.Timestamp) -> pd.Series:
    """Next calendar year by month if the café held its latest quarter's level (seasonally adjusted).

    Classical decomposition: seasonal index = average ratio of each month to its centred 12-month
    moving average; the last three months divided by their index give today's monthly level.
    """
    m = sd.monthly_clean(origin)
    ratio = (m / m.rolling(12).mean().rolling(2).mean().shift(-6)).dropna()
    s = ratio.groupby(ratio.index.month).mean()
    s = s / s.mean()
    last = origin.to_period("M")
    q = m.loc[last - 2:last]
    level = q.sum() / s.loc[[p.month for p in q.index]].sum()
    months = pd.period_range(f"{origin.year + 1}-01", f"{origin.year + 1}-12", freq="M")
    return pd.Series([level * s[p.month] for p in months], index=months)


def collect() -> dict:
    raw = load_daily()
    stores = load_stores(LIVE_END.strftime("%Y-%m-%d"))
    ann = pd.read_csv(OUTPUTS / "forecast_annual.csv").set_index("store")
    by_store = pd.read_csv(OUTPUTS / "forecast_by_store_month.csv")
    by_store["month"] = pd.PeriodIndex(by_store.month, freq="M")
    budget = pd.read_csv(DATA / "cfo_budget_2027.csv")
    comp = pd.read_csv(DATA / "cfo_budget_2027_comp_trend.csv")
    configs = load_configs()
    hold_p = pd.read_csv(OUTPUTS / "hold_scenario_2027.csv")  # src/hold_scenario.py
    hold_p["month"] = pd.PeriodIndex(hold_p.month, freq="M")
    hold_p = hold_p.set_index(["store", "month"]).hold
    hold_c = {s: hold_path(stores[s], ORIGIN) for s in stores}
    raw_m = raw.assign(month=raw.date.dt.to_period("M")).groupby(["store", "month"]).net_sales.sum()

    existing = budget[budget.status == "existing"]
    nine = existing.store.tolist()  # budget order
    order = nine + ["La Jolla Cove", "Nautilus"]
    bud = budget.dropna(subset=["store"]).set_index("store")

    cafes = []
    for s in order:
        a = ann.loc[s]
        ytd26, c26 = window(raw, s, "2026-01-01", "2026-09-30")
        ytd25, c25 = window(raw, s, "2025-01-01", "2025-09-30")
        oct26, oc26 = window(raw, s, "2026-10-01", "2026-10-14")
        oct25, oc25 = window(raw, s, "2025-10-01", "2025-10-14")
        q4_25, _ = window(raw, s, "2025-10-01", "2025-12-31")
        q_g = {}
        for qn, (qa, qz) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30")], start=1):
            cur, _ = window(raw, s, f"2026-{qa}", f"2026-{qz}")
            prev, _ = window(raw, s, f"2025-{qa}", f"2025-{qz}")
            q_g[qn] = cur / prev - 1
        e26 = a[f"{BLEND} 2026E"]
        model27 = [a[f"{m} 2027"] for m in MODELS]
        yoy = quarterly_yoy(raw, s)
        months26 = raw_m.loc[s].loc[pd.Period("2026-01", "M"):pd.Period("2026-09", "M")]
        full24 = pd.Timestamp(STORE_START[s]) <= pd.Timestamp("2024-01-01")
        c = {
            "key": s, "name": bud.budget_name.get(s, s) if s in bud.index else s,
            "short": SHORT.get(s, s), "verdict": VERDICT[s],
            "fy24": a.actual_2024, "fy25": a.actual_2025, "g25": a.actual_2025 / a.actual_2024 - 1 if full24 else None,
            "ytd26": ytd26, "ytd_g": ytd26 / ytd25 - 1, "chk_g": c26 / c25 - 1,
            "tkt_g": (ytd26 / c26) / (ytd25 / c25) - 1,
            "q1": q_g[1], "q2": q_g[2], "q3": q_g[3],
            "oct_sales": oct26, "oct_g": oct26 / oct25 - 1, "oct_chk_g": oc26 / oc25 - 1,
            "q4_25": q4_25, "q4_method": q4_25 * (1 + q_g[3]),
            "e26": e26, "q4_o": e26 - ytd26,
            "plan27": a[f"{PLAN} 2027"], "best27": a[f"{BLEND} 2027"], "low27": min(model27), "high27": max(model27),
            "quarters": [(qlabel(p), v) for p, v in yoy.items()],
            "slowest": yoy.min(), "slowest_q": qlabel(yoy.idxmin()), "first_q": qlabel(yoy.index[0]),
            "qmin": yoy.min(), "qmax": yoy.max(),
            "avg_m": months26.mean(), "min_m": months26.min(), "max_m": months26.max(),
            "best_m": raw_m.loc[s].max(), "best_m_label": raw_m.loc[s].idxmax().strftime("%b %Y"),
            "hold_c": hold_c[s].sum(), "hold_p": hold_p.loc[s].sum(),
            "yoy_mar": raw_m[(s, pd.Period("2026-03", "M"))] / raw_m[(s, pd.Period("2025-03", "M"))] - 1,
        }
        c["pg"] = c["plan27"] / e26 - 1
        c["lowg"], c["highg"] = c["low27"] / e26 - 1, c["high27"] / e26 - 1
        c["hold_lo"], c["hold_hi"] = sorted([c["hold_c"] / e26 - 1, c["hold_p"] / e26 - 1])
        c["added26"] = e26 - c["fy25"]
        c["added27_p"] = c["plan27"] - e26
        if s in bud.index:
            b = bud.loc[s]
            c.update(b26=b.est_2026, b27=b.objective_2027, bg=b.objective_2027 / b.est_2026 - 1,
                     key_assumption=b.key_assumption)
            c["q4_b"] = c["b26"] - ytd26
            c["d26"] = e26 / c["b26"] - 1
            c["gap"] = c["plan27"] - c["b27"]
            c["gap_low"] = c["low27"] - c["b27"]
            c["added27_b"] = c["b27"] - c["b26"]
        else:
            c.update(b26=None, b27=None, bg=None, key_assumption=None, q4_b=None, d26=None, gap=None,
                     gap_low=None, added27_b=None)
        cafes.append(c)
    C = {c["key"]: c for c in cafes}
    n = {"cafes": cafes, "C": C, "nine": nine, "order": order}

    # ---- totals for the nine existing cafés --------------------------------------------------
    nc = [C[s] for s in nine]
    tot = lambda k: sum(c[k] for c in nc)  # noqa: E731
    b26, b27, e26 = existing.est_2026.sum(), existing.objective_2027.sum(), tot("e26")
    plan27, low27, best27 = tot("plan27"), tot("low27"), tot("best27")
    n["t"] = {
        "b26": b26, "b27": b27, "bg": b27 / b26 - 1, "e26": e26, "d26": e26 / b26 - 1,
        "plan27": plan27, "pg": plan27 / e26 - 1, "low27": low27, "lowg": low27 / e26 - 1,
        "best27": best27, "high27": tot("high27"), "gap": plan27 - b27, "gap_low": low27 - b27,
        "fy25": tot("fy25"), "g26": e26 / tot("fy25") - 1, "added26": e26 - tot("fy25"), "added27_b": b27 - b26,
        "added27_p": plan27 - e26, "ytd26": tot("ytd26"),
        "q4_b": tot("q4_b"), "q4_o": tot("q4_o"), "q4_method": tot("q4_method"), "q4_25": tot("q4_25"),
    }
    t = n["t"]
    s26, c26 = window(raw, nine, "2026-01-01", "2026-09-30")
    s25, c25 = window(raw, nine, "2025-01-01", "2025-09-30")
    o26, _ = window(raw, nine, "2026-10-01", "2026-10-14")
    o25, _ = window(raw, nine, "2025-10-01", "2025-10-14")
    q326, _ = window(raw, nine, "2026-07-01", "2026-09-30")
    q325, _ = window(raw, nine, "2025-07-01", "2025-09-30")
    t.update(ytd_g=s26 / s25 - 1, chk_g=c26 / c25 - 1, tkt_g=(s26 / c26) / (s25 / c25) - 1,
             oct_sales=o26, oct_g=o26 / o25 - 1, q3_g=q326 / q325 - 1)
    t["booked"] = t["ytd26"] + o26
    t["booked_share"] = t["booked"] / b26
    t["to_come"] = b26 - t["booked"]
    t["q4_oct_pace"] = t["q4_25"] * (1 + t["oct_g"])
    t["hold_c"] = sum(c["hold_c"] for c in nc)
    t["hold_p"] = sum(c["hold_p"] for c in nc)
    t["hold_lo"], t["hold_hi"] = sorted([t["hold_c"] / e26 - 1, t["hold_p"] / e26 - 1])

    # ---- 2026 by month (nine cafés): booked vs still to come at the budget's finish -----------
    months = pd.period_range("2026-01", "2026-12", freq="M")
    booked = [raw_m.loc[nine].groupby(level="month").sum().get(m, 0.0) for m in months[:9]]
    q4_fc = by_store[by_store.store.isin(nine) & (by_store.month <= pd.Period("2026-12", "M"))]
    profile = q4_fc.groupby("month")[BLEND].sum()
    q4_budget_m = profile / profile.sum() * t["q4_b"]
    booked += [o26, 0.0, 0.0]
    to_come = [0.0] * 9 + [q4_budget_m.iloc[0] - o26, q4_budget_m.iloc[1], q4_budget_m.iloc[2]]
    n["booked_months"] = {"months": [m.to_timestamp() for m in months], "booked": booked, "to_come": to_come}

    # ---- 2026 bridge: budget -> Toast-based finish for all eleven cafés -----------------------
    cove, naut = C["La Jolla Cove"], C["Nautilus"]
    all26 = sum(c["e26"] for c in cafes)
    n["bridge26"] = {
        "budget9": b26, "cove": cove["b26"], "budget_total": b26 + cove["b26"], "nautilus": naut["e26"],
        "diff": all26 - (b26 + cove["b26"]) - naut["e26"], "ours": all26,
    }

    # ---- nine cafés by month: history, Q4 estimate, 2027 plan / budget / no-growth band -------
    act = raw_m.loc[nine].groupby(level="month").sum().loc[pd.Period("2025-01", "M"):pd.Period("2026-09", "M")]
    f9 = by_store[by_store.store.isin(nine)].groupby("month")[[BLEND, PLAN]].sum()
    est = f9.loc[pd.Period("2026-10", "M"):pd.Period("2026-12", "M"), BLEND]
    m27 = pd.period_range("2027-01", "2027-12", freq="M")
    months26 = pd.concat([act.loc[pd.Period("2026-01", "M"):], est])
    hc = sum(hold_c[s_] for s_ in nine)
    hp = hold_p.loc[nine].groupby(level="month").sum().reindex(m27)
    n["brand"] = {"act": act, "est": est, "plan": f9.loc[m27, PLAN], "budget": months26.to_numpy() * (b27 / e26),
                  "hold_lo": np.minimum(hc.to_numpy(), hp.to_numpy()),
                  "hold_hi": np.maximum(hc.to_numpy(), hp.to_numpy())}
    n["comp"] = comp

    # ---- track record: Jan-Sep 2026 forecast made in Sep 2025 ---------------------------------
    hold = pd.read_csv(OUTPUTS / "tuning" / "adopted_forecasts.csv")
    h = hold[(hold.origin == "2025-09-30") & hold.month.between("2026-01", "2026-09")]
    n["track"] = {"bias": h.f.sum() / h.actual.sum() - 1, "cafes": h.store.nunique()}
    n["plan_cfg"] = configs["plan"]

    # ---- new cafés: our last four openings, at 2026 prices -----------------------------------
    r = raw[raw.store.isin(ORIGINAL) & (raw.date.dt.month <= 9)]
    g = r.groupby(r.date.dt.year)[["net_sales", "checks"]].sum()
    avg_check = g.net_sales / g.checks
    price_idx = avg_check / avg_check.loc[2026]
    first = raw[raw.net_sales > 500].groupby("store").date.min()
    analogs = []
    for s in ANALOGS:
        d = raw[raw.store == s]
        row = {"store": SHORT.get(s, s), "opened": first[s]}
        for k in (8, 10, 12):
            w = d[(d.date >= first[s]) & (d.date < first[s] + pd.DateOffset(months=k))]
            row[k] = (w.net_sales / w.date.dt.year.map(price_idx)).sum()
            row[f"{k}_raw"] = w.net_sales.sum()
        analogs.append(row)
    avg = {k: np.mean([a[k] for a in analogs]) for k in (8, 10, 12)}
    new = []
    for _, b in budget[budget.status == "new"].iterrows():
        if pd.isna(b.objective_2027):
            continue
        opens = pd.Timestamp(b.opens)
        months_open = 13 - opens.month
        la = [a[months_open] for a in analogs if a["store"] in ("Beverly Hills", "Brentwood")]
        new.append({"name": b.budget_name, "opens": opens, "months": months_open, "budget": b.objective_2027,
                    "first12": b.key_assumption, "track": avg[months_open],
                    "best": max(a[months_open] for a in analogs), "la": np.mean(la)})
    tbd = budget[(budget.status == "new") & budget.objective_2027.isna() & (budget.opens.str.len() > 4)]
    n["openings"] = {"analogs": analogs, "avg": avg, "new": new, "tbd": tbd.budget_name.tolist(),
                     "best_name": max(analogs, key=lambda a: a[12])["store"],
                     "check_lift_23": 1 / price_idx.loc[2023] - 1, "check_lift_24": 1 / price_idx.loc[2024] - 1}

    # La Jolla Cove: the budget = Jan-Sep 2026 held flat + $125K a month for Oct-Dec 2027
    cv = by_store[by_store.store == "La Jolla Cove"].set_index("month")[PLAN]
    cove_q4_b = 3 * 125_000
    cove.update(b_js=cove["b27"] - cove_q4_b, b_q4=cove_q4_b,
                p_js=cv.loc[pd.Period("2027-01", "M"):pd.Period("2027-09", "M")].sum(),
                p_q4=cv.loc[pd.Period("2027-10", "M"):pd.Period("2027-12", "M")].sum())
    assert abs(cove["b_js"] - cove["ytd26"]) < 1_000, "budget Cove Jan-Sep should equal 2026 Jan-Sep actual"

    # ---- total company 2027 ---------------------------------------------------------------
    new_b = sum(x["budget"] for x in new)
    new_t = sum(x["track"] for x in new)
    total_b = b27 + cove["b27"] + new_b
    n["bridge27"] = {
        "budget": total_b, "existing": t["gap"], "nautilus": naut["plan27"], "cove": cove["plan27"] - cove["b27"],
        "new": new_t - new_b, "ours": plan27 + naut["plan27"] + cove["plan27"] + new_t, "new_budget": new_b,
        "new_track": new_t,
    }
    # ---- suggested objectives ---------------------------------------------------------------
    keep = {"defensible", "in_line"}
    for c in cafes:
        c["suggest"] = c["b27"] if c["verdict"] in keep else c["plan27"]
    sug9 = sum(C[s]["suggest"] for s in nine)
    n["suggest"] = {
        "nine": sug9, "nine_g": sug9 / e26 - 1, "nine_change": sug9 - b27,
        "raised": [C[s]["name"] for s in nine if C[s]["verdict"] not in keep],
        "kept": [C[s]["name"] for s in nine if C[s]["verdict"] in keep],
        "total": sug9 + naut["plan27"] + cove["plan27"] + new_t, "total_budget": total_b,
    }
    n["suggest"]["total_change"] = n["suggest"]["total"] - total_b
    n["suggest"]["total_g"] = n["suggest"]["total"] / all26 - 1
    n["all26"] = all26
    n["budget_total26"] = b26 + cove["b26"]

    # ---- where the gap sits -------------------------------------------------------------------
    strong = [c for c in nc if c["verdict"] == "too_low"]
    n["strong"] = {"names": [c["short"] for c in strong], "gap": sum(c["gap"] for c in strong),
                   "share": sum(c["gap"] for c in strong) / t["gap"],
                   "ytd_lo": min(c["ytd_g"] for c in strong), "ytd_hi": max(c["ytd_g"] for c in strong),
                   "bg_lo": min(c["bg"] for c in strong), "bg_hi": max(c["bg"] for c in strong)}
    n["below_low"] = [c["short"] for c in nc if c["gap_low"] > 0]
    n["below_slowest"] = [c["short"] for c in nc if c["bg"] < c["slowest"]]
    n["faster_than_budget"] = [c["short"] for c in nc if c["ytd_g"] > c["bg"]]
    n["first_day"], n["last_day"], n["store_days"] = raw.date.min(), raw.date.max(), len(raw)
    for c in cafes:
        c.update(story(c, n))
    check_verdicts(n)
    return n


def check_verdicts(n: dict) -> None:
    """Fail loudly if refreshed data no longer supports a verdict or a sentence on the café pages."""
    C = n["C"]
    for c in n["cafes"]:
        v, s = c["verdict"], c["key"]
        if v == "too_low":
            assert c["gap_low"] > 0 and c["gap"] / c["b27"] >= 0.05, s
        elif v == "slightly_low":
            assert c["gap"] > 0 and c["gap"] / c["b27"] < 0.05, s
        elif v == "defensible":
            assert c["low27"] <= c["b27"] <= c["high27"] or abs(c["hold_c"] / c["b27"] - 1) < 0.03, s
        elif v == "in_line":
            assert abs(c["gap"]) / c["b27"] < 0.02, s
        elif v == "above":
            assert c["gap"] < 0, s
        elif v == "missing":
            assert c["b27"] is None and c["oct_sales"] > 0, s
    nine = [C[s] for s in n["nine"]]
    assert max(nine, key=lambda c: c["gap"])["key"] == "Little Italy"  # "largest gap of any café"
    assert C["Carmel Valley"]["slowest"] > C["Carmel Valley"]["bg"]
    assert C["Newport"]["slowest"] > C["Newport"]["bg"]
    assert C["Carlsbad"]["bg"] < C["Carlsbad"]["slowest"] / 2.5  # "a third of its slowest quarter"
    assert C["Coronado"]["q1"] > C["Coronado"]["q2"] > C["Coronado"]["q3"]  # "cooled every quarter"
    assert C["La Jolla"]["low27"] > C["La Jolla"]["b27"]  # "all three of our models are above it"
    assert C["Brentwood"]["max_m"] / C["Brentwood"]["min_m"] < 1.15  # "held at about ... since January"
    li = C["Little Italy"]
    assert li["hold_lo"] - 0.01 <= li["bg"] <= li["hold_hi"] + 0.01  # "that is the budget's"
    assert n["t"]["hold_lo"] - 0.005 <= n["t"]["bg"] <= n["t"]["hold_hi"] + 0.005  # budget ~ no growth
    assert all(c["ytd_g"] > c["bg"] for c in nine)  # "every café is growing faster"


def story(c: dict, n: dict) -> dict:
    """Headlines and plain-English read for the café's two pages."""
    s = c["key"]
    o = {}
    if s == "Beverly Hills":
        o["h1"] = f"Beverly Hills: the budget adds {fk(c['added27_b'])} in 2027; the café added {fk(c['added26'])} in 2026"
        o["h2"] = (f"Still ramping, and faster: {pct0(c['q3'])} in Q3 and {pct0(c['oct_g'])} in early October. "
                   f"A {pct(c['bg'])} objective assumes the ramp stops in January")
        o["says"] = (f"Three years in, Beverly Hills is speeding up, not settling down: sales are up {apct(c['ytd_g'])} "
                     f"this year, {pct(c['q3'])} in Q3 and {pct(c['oct_g'])} in October 1–14. Its slowest quarter since "
                     f"its first anniversary was {pct(c['slowest'])} ({c['slowest_q']}). The budget agrees the ramp "
                     f"continues but prices it at {pct(c['bg'])}.")
        o["verdict_line"] = (f"Our plan: {fm(c['plan27'])} ({pct(c['pg'])}), {fk(c['gap'])} above the budget. "
                             f"Even our most cautious model is {fk(c['gap_low'])} above it.")
    elif s == "Brentwood":
        o["h1"] = f"Brentwood: the budget adds {fk(c['added27_b'])} in 2027; our models add {fk(c['added27_p'])}"
        o["h2"] = "Flat since January. On Brentwood, the budget's caution is fair"
        o["says"] = (f"Brentwood stepped up in late 2025 and has held at about {fk(c['avg_m'])} a month since "
                     f"January (between {fk(c['min_m'])} and {fk(c['max_m'])}). This year's {pct(c['ytd_g'])} growth is "
                     f"mostly that step being lapped: it fades to {pct(c['oct_g'])} in October 1–14, with checks "
                     f"{pct(c['oct_chk_g'])} on a year ago. If sales hold this level, 2027 lands close to 2026, which "
                     f"is where the budget sits. Our models ({pct(c['lowg'], 0)} to {pct(c['highg'], 0)}) still lean on "
                     f"the 2024–25 ramp.")
        o["verdict_line"] = (f"We'd keep the budget's {fm(c['b27'])}: it matches today's level. Our models' "
                             f"{fm(c['plan27'])} needs growth to resume, so we count it as upside, not plan.")
    elif s == "Carlsbad":
        o["h1"] = f"Carlsbad: the budget adds {fk(c['added27_b'])} in 2027; the café added {fk(c['added26'])} in 2026"
        o["h2"] = (f"Growing about 10% a year, quarter after quarter. The budget's {pct(c['bg'])} is about a third "
                   f"of its slowest quarter")
        o["says"] = (f"Carlsbad's growth has eased from the 20s in 2025 to around 10%, and it is holding there: "
                     f"{pct(c['q1'])}, {pct(c['q2'])} and {pct(c['q3'])} in the three quarters of 2026 and "
                     f"{pct(c['oct_g'])} in October 1–14, with checks {pct(c['oct_chk_g'])}. Its slowest quarter since "
                     f"its first anniversary was {pct(c['slowest'])} ({c['slowest_q']}). The budget calls it "
                     f"“steady growth” and sets {pct(c['bg'])}.")
        o["verdict_line"] = (f"Our plan: {fm(c['plan27'])} ({pct(c['pg'])}), {fk(c['gap'])} above the budget. "
                             f"Our most cautious model is {fk(c['gap_low'])} above it.")
    elif s == "Coronado":
        o["h1"] = f"Coronado: the budget's {pct(c['bg'])} sits inside our range, at the low end"
        o["h2"] = (f"Growth is cooling fast: {pct0(c['q1'])} in Q1, {pct0(c['q3'])} in Q3, "
                   f"{pct0(c['oct_g'])} in October")
        o["says"] = (f"Coronado had the strongest start to 2026 ({pct(c['q1'])} in Q1) and has cooled every quarter "
                     f"since: {pct(c['q2'])} in Q2, {pct(c['q3'])} in Q3 and {pct(c['oct_g'])} in October 1–14, with "
                     f"checks {pct(c['oct_chk_g'])}. Our three models span {fm(c['low27'])} to {fm(c['high27'])}, and "
                     f"the budget's {fm(c['b27'])} is inside that range. The budget's “moderating growth” is a fair "
                     f"call; if today's pace holds, our plan of {fm(c['plan27'])} ({pct(c['pg'])}) is the better number.")
        o["verdict_line"] = (f"We'd keep the budget's {fm(c['b27'])} and revisit after the first quarter; "
                             f"our plan is {fk(c['gap'])} higher.")
    elif s == "Carmel Valley":
        impact = 1 - c["b27"] / c["plan27"]
        o["h1"] = (f"Del Mar Café: the budget adds {fk(c['added27_b'])} in 2027; the café added "
                   f"{fk(c['added26'])} in 2026")
        o["h2"] = (f"No sign of the competitor yet: up {apct(c['oct_g'], 0)} in early October. A {pct(c['bg'])} budget "
                   f"books a {impact:.0%} hit before it happens")
        o["says"] = (f"Del Mar is our largest café and one of the steadiest: every quarter since late 2023 has grown "
                     f"at least {pct(c['slowest'])}, and October 1–14 is running {pct(c['oct_g'])} with checks "
                     f"{pct(c['oct_chk_g'])}. The budget's {pct(c['bg'])} assumes a competitor next door takes about "
                     f"{impact:.0%} of the sales our plan expects. That may happen, but nothing in the sales shows it yet.")
        o["verdict_line"] = (f"Our plan: {fm(c['plan27'])} ({pct(c['pg'])}), {fk(c['gap'])} above the budget. "
                             f"We'd show the competitor as its own risk line rather than lower the base.")
    elif s == "La Jolla":
        o["h1"] = f"La Jolla Café: the budget is {fk(c['gap'])} below our plan, a small gap"
        o["h2"] = (f"Growth picked back up after a soft spring: {pct0(c['q3'])} in Q3 and {pct0(c['oct_g'])} "
                   f"in early October")
        o["says"] = (f"La Jolla Café slowed in the first half ({pct(c['q1'])} in Q1, {pct(c['q2'])} in Q2) but has "
                     f"picked up again: {pct(c['q3'])} in Q3 and {pct(c['oct_g'])} in October 1–14. The budget's "
                     f"{pct(c['bg'])} matches the soft patch rather than the recovery. All three of our models are "
                     f"above it.")
        o["verdict_line"] = f"Our plan: {fm(c['plan27'])} ({pct(c['pg'])}), {fk(c['gap'])} above the budget."
    elif s == "La Jolla Juicery":
        o["h1"] = f"La Jolla Juicery: the budget and our plan agree at about {fm(c['plan27'])}"
        o["h2"] = f"Volatile but steady: our plan lands at {pct(c['pg'])}, the budget at {pct(c['bg'])}"
        o["says"] = (f"The Juicery's quarterly growth has swung between {pct(c['qmin'])} and {pct(c['qmax'])} since "
                     f"{c['first_q']}. It is up {apct(c['ytd_g'])} this year, but October 1–14 has cooled to "
                     f"{pct(c['oct_g'])}, with checks {pct(c['oct_chk_g'])}. Our plan of {fm(c['plan27'])} is within "
                     f"{fk(c['gap'])} of the budget.")
        o["verdict_line"] = f"Keep the budget's {fm(c['b27'])}."
    elif s == "Little Italy":
        mar = c["yoy_mar"]
        o["h1"] = (f"Little Italy: the budget adds {fk(c['added27_b'])} in 2027; the café added "
                   f"{fk(c['added26'])} in 2026")
        o["h2"] = (f"A step up in March that is still building ({pct0(c['oct_g'])} in early October). "
                   f"The budget's {pct(c['bg'])} is roughly today's level, frozen")
        o["says"] = (f"Little Italy stepped up in March 2026 (sales {pct(mar)} on March 2025) and has kept climbing: "
                     f"{pct(c['q3'])} in Q3 and {pct(c['oct_g'])} in October 1–14, almost all from more guests "
                     f"(checks {pct(c['chk_g'])} this year). Because January and February came before the step, "
                     f"simply holding today's level would grow 2027 by about {apct(c['hold_lo'], 0)}–"
                     f"{apct(c['hold_hi'], 0)}: that is the budget's {pct(c['bg'])}.")
        o["verdict_line"] = (f"Our plan: {fm(c['plan27'])} ({pct(c['pg'])}), {fk(c['gap'])} above the budget, "
                             f"the largest gap of any café.")
    elif s == "Newport":
        o["h1"] = f"Newport Beach: the budget adds {fk(c['added27_b'])} in 2027; our plan adds {fk(c['added27_p'])}"
        o["h2"] = f"Even its slowest quarter ({pct(c['slowest'])}) beats the budget's {pct(c['bg'])} for 2027"
        o["says"] = (f"After a fast second year, Newport has settled into steady growth: {pct(c['q1'])}, "
                     f"{pct(c['q2'])} and {pct(c['q3'])} in the three quarters of 2026 and {pct(c['oct_g'])} in "
                     f"October 1–14, carried by more guests (checks {pct(c['chk_g'])} this year). The budget's "
                     f"“steady growth” at {pct(c['bg'])} is below every quarter since Newport's first anniversary.")
        o["verdict_line"] = (f"Our plan: {fm(c['plan27'])} ({pct(c['pg'])}), {fk(c['gap'])} above the budget; "
                             f"even our most cautious model is {fk(c['gap_low'])} above it.")
    elif s == "La Jolla Cove":
        ratio = 125_000 / c["avg_m"]
        o["h1"] = f"La Jolla Cove: the budget is {fk(c['gap'])} above our plan, all of it from the remodel"
        o["h2"] = "Flat this year; the remodel would have to more than double monthly sales"
        o["says"] = (f"The Cove is {'down' if c['ytd_g'] < 0 else 'up'} {apct(c['ytd_g'])} this year and "
                     f"averages {fk(c['avg_m'])} a month; its best month ever was {fk(c['best_m'])} "
                     f"({c['best_m_label']}). The budget holds January–September 2027 flat to 2026, which matches "
                     f"our view, then assumes $125K a month for October–December 2027 after the remodel: "
                     f"{ratio:.1f}× today's average. We have no sales history that can test a remodel.")
        o["verdict_line"] = (f"We'd budget the Cove on its current trend ({fm(c['plan27'])}) and add the remodel "
                             f"lift once it reopens and trades.")
        o["ratio"] = ratio
    elif s == "Nautilus":
        o["h1"] = "Nautilus: still trading, but not in the budget"
        o["h2"] = f"Growing {apct(c['oct_g'], 0)} in October, and missing from both years of the budget"
        o["says"] = (f"Nautilus is our small weekday-only café. It grew {apct(c['ytd_g'])} in January–September, "
                     f"{pct(c['q3'])} in Q3 and {pct(c['oct_g'])} in October 1–14, when it took {fk(c['oct_sales'])}. "
                     f"It does not appear anywhere in the budget: not among the nine existing cafés and not under "
                     f"new and remodeled cafés.")
        o["verdict_line"] = (f"Confirm whether Nautilus is closing or reported elsewhere. If not, add {fm(c['e26'])} "
                             f"to 2026 and {fm(c['plan27'])} to 2027.")
    return o


# ------------------------------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------------------------------
def cafe_series(c: dict, raw_m: pd.Series, by_store: pd.DataFrame) -> dict:
    s = c["key"]
    start = max(pd.Period("2024-01", "M"), pd.Timestamp(STORE_START[s]).to_period("M")
                + (0 if pd.Timestamp(STORE_START[s]).day == 1 else 1))
    act = raw_m.loc[s].loc[start:pd.Period("2026-09", "M")]
    f = by_store[by_store.store == s].set_index("month")
    est = f.loc[pd.Period("2026-10", "M"):pd.Period("2026-12", "M"), BLEND]
    fc = f.loc[pd.Period("2027-01", "M"):pd.Period("2027-12", "M")]
    months26 = pd.concat([raw_m.loc[s].loc[pd.Period("2026-01", "M"):pd.Period("2026-09", "M")], est])
    budget = months26.to_numpy() * (c["b27"] / months26.sum()) if c["b27"] else None
    if s == "La Jolla Cove":
        budget = np.r_[months26.to_numpy()[:9], [c["b_q4"] / 3] * 3]
    return {"act": act, "est": est, "fc": fc, "budget": budget, "start": start}


def year_labels(c: dict, months: list[pd.Period]) -> list[tuple]:
    out = []
    for y in sorted({m.year for m in months}):
        idx = [i for i, m in enumerate(months) if m.year == y]
        lo, hi = idx[0] - 0.5, idx[-1] + 0.5
        if y == 2024:
            full = pd.Timestamp(STORE_START[c["key"]]) <= pd.Timestamp("2024-01-01")
            label = f"2024\n{fm(c['fy24'])}" if full else f"2024\nfrom {months[idx[0]].strftime('%b')}"
        elif y == 2025:
            label = f"2025\n{fm(c['fy25'])}"
        elif y == 2026:
            label = f"2026E\n{fm(c['e26'])}"
        else:
            label = (f"2027\nplan {fm(c['plan27'])} · budget {fm(c['b27'])}" if c["b27"]
                     else f"2027\nplan {fm(c['plan27'])}")
        out.append((lo, hi, label))
    return out


def make_charts(n: dict) -> dict:
    raw = load_daily()
    raw_m = raw.assign(month=raw.date.dt.to_period("M")).groupby(["store", "month"]).net_sales.sum()
    by_store = pd.read_csv(OUTPUTS / "forecast_by_store_month.csv")
    by_store["month"] = pd.PeriodIndex(by_store.month, freq="M")
    out = {}
    bm = n["booked_months"]
    out["booked"] = dc.booked_bars(bm["months"], bm["booked"], bm["to_come"], size=(7.4, 3.5))
    t = n["t"]
    comp = n["comp"]
    out["comp"] = dc.growth_by_year(
        ["2024", "2025", "2026E", "2027\nbudget", "2027\nour plan"],
        list(comp.same_store_growth.iloc[:3]) + [t["bg"], t["pg"]],
        [charts.INK, charts.INK, charts.INK, dc.BUDGET, charts.PINK], size=(4.3, 2.5))
    b = n["brand"]
    out["brand"] = dc.brand_monthly(
        [m.to_timestamp() for m in b["act"].index], b["act"].to_numpy(),
        [m.to_timestamp() for m in b["est"].index], b["est"].to_numpy(),
        [m.to_timestamp() for m in b["plan"].index], b["plan"].to_numpy(), b["budget"], b["hold_lo"], b["hold_hi"],
        size=(7.4, 3.3))
    nine = sorted([n["C"][s] for s in n["nine"]], key=lambda c: -c["ytd_g"])
    out["dumbbell"] = dc.cafe_dumbbell([c["short"] for c in nine], [c["bg"] for c in nine],
                                       [c["ytd_g"] for c in nine], [c["pg"] for c in nine], size=(7.2, 3.9))
    gaps = sorted([n["C"][s] for s in n["nine"]], key=lambda c: -c["gap"])
    color = {"too_low": charts.PINK, "slightly_low": "#f59ab8", "defensible": charts.G400, "in_line": charts.G400}
    out["gaps"] = dc.gap_bars([c["short"] for c in gaps], [c["gap"] for c in gaps],
                              [color[c["verdict"]] for c in gaps], size=(5.2, 4.3))
    out["cafe"] = {}
    for c in n["cafes"]:
        sr = cafe_series(c, raw_m, by_store)
        fc = sr["fc"]
        months = list(sr["act"].index) + list(sr["est"].index) + list(fc.index)
        monthly = dc.cafe_monthly(
            [m.to_timestamp() for m in sr["act"].index], sr["act"].to_numpy(),
            [m.to_timestamp() for m in sr["est"].index], sr["est"].to_numpy(),
            [m.to_timestamp() for m in fc.index], fc[PLAN].to_numpy(), fc[MODELS].min(axis=1).to_numpy(),
            fc[MODELS].max(axis=1).to_numpy(),
            sr["budget"],
            year_labels(c, months), size=(7.5, 4.2), name=f"m-{c['key']}")
        q = dc.quarterly_growth([lab.replace(" 20", " ’") for lab, _ in c["quarters"]], [v for _, v in c["quarters"]],
                                ("Oct 1–14", c["oct_g"]), c["bg"], c["pg"], size=(7.0, 4.3), name=f"q-{c['key']}")
        out["cafe"][c["key"]] = {"monthly": monthly, "quarterly": q}
    return out


# ------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------
def export_csv(n: dict) -> None:
    cols = ["name", "verdict", "fy25", "ytd_g", "q3", "oct_g", "chk_g", "tkt_g", "b26", "e26", "d26", "q4_b", "q4_o",
            "b27", "bg", "low27", "plan27", "best27", "high27", "pg", "gap", "gap_low", "hold_c", "hold_p", "suggest",
            "key_assumption"]
    df = pd.DataFrame([{k: c[k] for k in cols} for c in n["cafes"]])
    df["verdict"] = df.verdict.map(VERDICT_LABEL)
    df = df.rename(columns={
        "name": "cafe", "fy25": "actual_2025", "ytd_g": "growth_jan_sep_2026", "q3": "growth_q3_2026",
        "oct_g": "growth_oct_1_14_2026", "chk_g": "checks_growth_jan_sep_2026", "tkt_g": "avg_check_growth_jan_sep_2026",
        "b26": "budget_2026_finish", "e26": "toast_2026_finish", "d26": "toast_vs_budget_2026",
        "q4_b": "budget_oct_dec_2026", "q4_o": "our_oct_dec_2026", "b27": "budget_2027", "bg": "budget_growth_2027",
        "low27": "lowest_model_2027", "plan27": "our_plan_2027", "best27": "best_estimate_2027",
        "high27": "highest_model_2027", "pg": "our_plan_growth_2027", "gap": "plan_minus_budget_2027",
        "gap_low": "lowest_model_minus_budget_2027", "hold_c": "no_growth_2027_seasonal_runrate",
        "hold_p": "no_growth_2027_prophet_frozen_trend",
        "suggest": "suggested_2027", "key_assumption": "budget_key_assumption"})
    df.to_csv(CSV, index=False, float_format="%.4f")


def render(n: dict, svgs: dict) -> Path:
    env = Environment(loader=FileSystemLoader(ROOT / "report"), autoescape=False)
    env.filters.update(fm=fm, fk=fk, sfk=sfk, sfm=sfm, pct=pct, pct0=pct0, apct=apct, frange=frange, words=words)
    html = env.get_template("deck.html.j2").render(n=n, svg=svgs, VERDICT_LABEL=VERDICT_LABEL)
    path = BUILD / "parakeet_2027_budget_review.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", help="path to the parakeet-cafe-design skill folder")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--shots", help="also write page PNGs to this folder (visual QA)")
    args = ap.parse_args()
    ds = find_design_system(args.ds)
    charts.setup(copy_assets(ds))
    for f in PHOTOS:
        shutil.copy(ds / "assets" / f, BUILD / "assets" / Path(f).name)
    n = collect()
    export_csv(n)
    html = render(n, make_charts(n))
    print(f"HTML -> {html.relative_to(ROOT)}")
    print(f"CSV  -> {CSV.relative_to(ROOT)}")
    if not args.no_pdf:
        node_path = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
        env = {**os.environ, "NODE_PATH": node_path}
        cmd = ["node", str(ROOT / "report" / "render_pdf.cjs"), str(html), str(PDF), args.shots or "",
               "13.333in", "7.5in"]
        subprocess.run(cmd, check=True, env=env)
        print(f"PDF  -> {PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
