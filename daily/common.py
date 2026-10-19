"""Shared pieces for the daily-shape study: the daily panel, special days, seasons, the methods that split a month
into days, and the metric.

The monthly totals are fixed (they were sent), so every method here only has to say how a month divides into its
days. The metric does exactly that: take each actual month, split it with the method's daily weights, and compare
with the actual days (daily WAPE given the true month).

Reads the repository's data and settings (src/forecasting.py) without changing them.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting import HOLIDAYS, PROPHET_DEFAULT, load_configs, load_stores  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
LAST_DAY = "2026-10-14"
STORES = ["Beverly Hills", "Brentwood", "Carlsbad", "Coronado", "Carmel Valley", "La Jolla", "La Jolla Juicery",
          "Little Italy", "Newport", "La Jolla Cove"]  # the ten stores of the targets (Nautilus left out)
NAME = {"Carmel Valley": "Del Mar Café", "La Jolla": "La Jolla Café", "Little Italy": "Little Italy Café",
        "Newport": "Newport Beach"}
SEASONS = ["term", "summer", "holidays", "spring"]
PROPHET_CFG = {**PROPHET_DEFAULT, **load_configs()["models"]["Prophet"]}


# ------------------------------------------------------------------------------------------
# Calendar: special days and seasons
# ------------------------------------------------------------------------------------------
def special_days(years=range(2022, 2029)) -> pd.Series:
    """One label per special date: the holiday table, with an observed holiday kept only where no named holiday
    falls, plus the days between Christmas and New Year's Eve."""
    h = HOLIDAYS.copy()
    h["named"] = h.holiday != "Observed holiday"
    h = h.sort_values(["ds", "named"], ascending=[True, False]).drop_duplicates("ds")
    out = pd.Series(h.holiday.to_numpy(), index=pd.DatetimeIndex(h.ds))
    extra = {}
    for y in years:
        extra[pd.Timestamp(y, 12, 26)] = "Dec 26"
        for d in range(27, 31):
            extra[pd.Timestamp(y, 12, d)] = "Holiday week"
    for d, lab in extra.items():
        if d not in out.index:
            out.loc[d] = lab
    return out.sort_index()


SPECIAL = special_days()
EASTER = {d.year: d for d in SPECIAL[SPECIAL == "Easter"].index}


def season_of(dates: pd.DatetimeIndex) -> np.ndarray:
    """School-calendar seasons (a priori): summer Jun 15 - Aug 20, holiday season Nov 22 - Jan 5, spring break the
    week either side of Easter, school term the rest."""
    dates = pd.DatetimeIndex(dates)
    md = dates.month * 100 + dates.day
    s = np.full(len(dates), "term", dtype=object)
    s[(md >= 615) & (md <= 820)] = "summer"
    s[(md >= 1122) | (md <= 105)] = "holidays"
    easter = pd.DatetimeIndex([EASTER[y] for y in dates.year])
    s[(dates >= easter - pd.Timedelta(days=7)) & (dates <= easter + pd.Timedelta(days=7))] = "spring"
    return s


# ------------------------------------------------------------------------------------------
# The daily panel
# ------------------------------------------------------------------------------------------
def build_panel(last_day: str = LAST_DAY) -> pd.DataFrame:
    """Every store-day from the store's first full month: sales, the trading calendar, cleaning flags and the
    calendar features the methods use."""
    stores = load_stores(last_day)
    rows = []
    for s in STORES:
        sd = stores[s]
        d = sd.daily.loc[pd.Timestamp(sd.first_full_month.start_time):last_day]
        rows.append(pd.DataFrame({"store": s, "date": d.index, "sales": d.sales.fillna(0).to_numpy(),
                                  "open": d.open.to_numpy(), "anomaly": d.anomaly.astype(bool).to_numpy()}))
    p = pd.concat(rows, ignore_index=True)
    return add_calendar(p)


def add_calendar(p: pd.DataFrame) -> pd.DataFrame:
    dt = pd.DatetimeIndex(p.date)
    p["dow"] = dt.dayofweek
    p["month"] = dt.to_period("M")
    p["season"] = season_of(dt)
    p["special"] = SPECIAL.reindex(dt).fillna("").to_numpy()
    p["doy"] = dt.dayofyear
    p["dom"] = dt.day
    p["normal"] = (p.open == 1) & ~p.anomaly & (p.sales > 0)
    return p


def future_panel(store: str, dates: pd.DatetimeIndex, open_flags: np.ndarray) -> pd.DataFrame:
    p = pd.DataFrame({"store": store, "date": dates, "sales": np.nan, "open": open_flags, "anomaly": False})
    return add_calendar(p)


# ------------------------------------------------------------------------------------------
# The metric
# ------------------------------------------------------------------------------------------
def shape_errors(p: pd.DataFrame, col: str) -> pd.DataFrame:
    """Daily error given the true month, per store-month: split each actual month (its normal trading days) with the
    method's weights and compare day by day. Returns the absolute error and the actual sales per store-month."""
    z = p[p.normal & (p[col] > 0)].copy()
    tot = z.groupby(["store", "month"]).sales.transform("sum")
    share = z[col] / z.groupby(["store", "month"])[col].transform("sum")
    z["err"] = (z.sales - share * tot).abs()
    z["pred"] = share * tot
    return z


def wape(z: pd.DataFrame) -> float:
    return z.err.sum() / z.sales.sum()


# ------------------------------------------------------------------------------------------
# Methods: daily weights for target dates, using only data up to the origin
# ------------------------------------------------------------------------------------------
def m_prophet(sd, origin: pd.Timestamp, dates: pd.DatetimeIndex, free_weekly: bool = False,
              summer_weekly: bool = False) -> np.ndarray:
    """Prophet with the adopted settings (as the run-rate and the current daily file use it), or with the weekly
    pattern freed from the tuned prior, optionally a separate weekly pattern in summer."""
    from prophet import Prophet

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    c = PROPHET_CFG
    m = Prophet(growth="linear", seasonality_mode=c["seasonality_mode"], yearly_seasonality=int(c["yearly_fourier"]),
                weekly_seasonality=not (free_weekly or summer_weekly), daily_seasonality=False, holidays=HOLIDAYS,
                changepoint_prior_scale=c["changepoint_prior_scale"],
                seasonality_prior_scale=c["seasonality_prior_scale"],
                holidays_prior_scale=c["holidays_prior_scale"], changepoint_range=c["changepoint_range"],
                uncertainty_samples=0)
    train = sd.train_days(origin)
    if summer_weekly:
        m.add_seasonality("weekly_summer", period=7, fourier_order=3, prior_scale=10.0, condition_name="is_summer")
        m.add_seasonality("weekly_rest", period=7, fourier_order=3, prior_scale=10.0, condition_name="not_summer")
        for df in (train,):
            df["is_summer"] = season_of(df.ds) == "summer"
            df["not_summer"] = ~df.is_summer
    elif free_weekly:
        m.add_seasonality("weekly", period=7, fourier_order=3, prior_scale=10.0)
    m.fit(train)
    fut = pd.DataFrame({"ds": dates})
    if summer_weekly:
        fut["is_summer"] = season_of(fut.ds) == "summer"
        fut["not_summer"] = ~fut.is_summer
    f = 1 + m.predict(fut).multiplicative_terms.to_numpy()
    return np.clip(f, 0.05, None) * sd.open_flags(dates)


def weekday_profiles(hist: pd.DataFrame, origin: pd.Timestamp) -> dict[str, np.ndarray]:
    """Log weekday pattern by season: the last 52 weeks' pattern, plus each season's usual deviation from the
    store's own yearly pattern (learned from all earlier full weeks, net of year-to-year level changes)."""
    h = hist[hist.normal & (hist.special == "")].copy()
    h["week"] = h.date - pd.to_timedelta(h.dow, unit="D")
    full = h.groupby("week").dow.transform("count") == 7
    h = h[full]
    h["ls"] = np.log(h.sales) - np.log(h.groupby("week").sales.transform("mean"))
    h["wseason"] = season_of(h.week + pd.Timedelta(days=3))
    last = h[h.week > origin - pd.Timedelta(weeks=52)]
    base = last.groupby("dow").ls.mean().reindex(range(7)).fillna(0).to_numpy()
    base = base - base.mean()
    h["year"] = (h.week + pd.Timedelta(days=3)).dt.year
    yr = h.groupby(["year", "dow"]).ls.transform("mean")
    dev = h.assign(dev=h.ls - yr).groupby(["wseason", "dow"]).dev.agg(["mean", "count"])
    out = {"base": base}
    for s in SEASONS:
        if s in dev.index.get_level_values(0):
            d = dev.loc[s].reindex(range(7))
            k = d["count"].fillna(0).to_numpy()
            m = d["mean"].fillna(0).to_numpy() * k / (k + 4)  # shrink seasons seen in few weeks
            out[s] = base + (m - m.mean())
        else:
            out[s] = base
    return out


def special_effects(panel: pd.DataFrame, origin: pd.Timestamp, shrink: float = 2.0) -> pd.DataFrame:
    """Each special day's effect per store: log of the day's sales against the same weekday one and two weeks
    before and after (normal days only, all before the origin), averaged over its occurrences and shrunk toward the
    average across stores."""
    p = panel[(panel.date <= origin)].set_index(["store", "date"])
    sales = p.sales.where(p.normal & (p.special == ""))
    rows = []
    for (s, d), r in p[(p.special != "") & p.normal].iterrows():
        refs = [sales.get((s, d + pd.Timedelta(weeks=k))) for k in (-2, -1, 1, 2)]
        refs = [v for v in refs if v is not None and not np.isnan(v)]
        if len(refs) >= 2:
            rows.append({"store": s, "special": r.special, "eff": np.log(r.sales / np.mean(refs))})
    e = pd.DataFrame(rows)
    if e.empty:
        return pd.DataFrame(columns=["store", "special", "eff"])
    pooled = e.groupby("special").eff.mean()
    g = e.groupby(["store", "special"]).eff.agg(["mean", "count"]).reset_index()
    g["eff"] = (g["mean"] * g["count"] + pooled.reindex(g.special).to_numpy() * shrink) / (g["count"] + shrink)
    # stores that never saw a special day get the pooled effect
    allp = pd.MultiIndex.from_product([STORES, pooled.index], names=["store", "special"]).to_frame(index=False)
    allp = allp.merge(g[["store", "special", "eff"]], how="left", on=["store", "special"])
    allp["eff"] = allp.eff.fillna(allp.special.map(pooled))
    return allp


def day_of_year_curve(panel: pd.DataFrame, origin: pd.Timestamp, profiles: dict, effects: pd.DataFrame,
                      bandwidth: float = 6.0) -> np.ndarray:
    """Smooth level by day of year within months (e.g. the second half of June once school is out), pooled across
    stores: what is left of each normal day after its month's mean, weekday pattern and special-day effect,
    smoothed over the calendar."""
    h = panel[panel.normal & (panel.date <= origin)].copy()
    h["ls"] = np.log(h.sales)
    wd = np.array([profiles[r.store][r.season][r.dow] for r in h[["store", "season", "dow"]].itertuples()])
    eff = h.merge(effects, how="left", on=["store", "special"]).eff.fillna(0).to_numpy()
    h["r"] = h.ls - wd - eff
    h["r"] -= h.groupby(["store", "month"]).r.transform("mean")
    by = h.groupby("doy").r.agg(["sum", "count"]).reindex(range(1, 367), fill_value=0)
    x = np.arange(1, 367)
    curve = np.zeros(366)
    for i, d in enumerate(x):
        dist = np.minimum(np.abs(x - d), 366 - np.abs(x - d))
        k = np.exp(-0.5 * (dist / bandwidth) ** 2)
        curve[i] = (k * by["sum"]).sum() / max((k * by["count"]).sum(), 1e-9)
    return curve


class Structural:
    """Weekday pattern by season (recent level, seasonal swing from earlier years), calibrated special days and a
    smooth day-of-year level, all in logs: weight = exp(weekday + special + day of year), zero when closed."""

    def __init__(self, panel: pd.DataFrame, origin: pd.Timestamp, seasonal: bool = True, specials: bool = True,
                 use_curve: bool = True):
        hist = panel[panel.date <= origin]
        self.seasonal, self.specials = seasonal, specials
        self.profiles = {s: weekday_profiles(hist[hist.store == s], origin) for s in STORES if (hist.store == s).any()}
        self.effects = special_effects(panel, origin)
        self.curve = day_of_year_curve(panel, origin, self.profiles, self.effects) if use_curve else np.zeros(366)

    def weights(self, fp: pd.DataFrame) -> np.ndarray:
        wd = np.array([self.profiles[r.store][r.season if self.seasonal else "base"][r.dow]
                       for r in fp[["store", "season", "dow"]].itertuples()])
        eff = fp[["store", "special"]].merge(self.effects, how="left", on=["store", "special"]).eff.fillna(0).to_numpy()
        return np.exp(wd + eff * self.specials + self.curve[fp.doy.to_numpy() - 1]) * fp.open.to_numpy()


LGB_FEATURES = ["store_code", "dow", "season_code", "special_code", "doy", "dom", "days_left", "to_xmas", "to_thanks",
                "to_easter", "to_july4"]


def lgb_features(p: pd.DataFrame) -> pd.DataFrame:
    dt = pd.DatetimeIndex(p.date)
    f = pd.DataFrame(index=p.index)
    f["store_code"] = pd.Categorical(p.store, categories=STORES).codes
    f["dow"] = p.dow.to_numpy()
    f["season_code"] = pd.Categorical(p.season, categories=SEASONS).codes
    labels = sorted(set(SPECIAL.unique()))
    f["special_code"] = pd.Categorical(p.special.replace("", "none"), categories=["none"] + labels).codes
    f["doy"] = p.doy.to_numpy()
    f["dom"] = p.dom.to_numpy()
    f["days_left"] = (dt.days_in_month - dt.day).to_numpy()

    def to(name):
        days = SPECIAL[SPECIAL == name].index
        idx = np.searchsorted(days.values, dt.values)
        nxt = days[np.minimum(idx, len(days) - 1)]
        prv = days[np.maximum(idx - 1, 0)]
        a, b = (dt - prv).days.to_numpy(), (nxt - dt).days.to_numpy()
        v = np.where(np.abs(a) < np.abs(b), a, -b)
        return np.clip(v, -10, 10)
    f["to_xmas"], f["to_thanks"] = to("Christmas Day"), to("Thanksgiving Day")
    f["to_easter"], f["to_july4"] = to("Easter"), to("Independence Day")
    return f[LGB_FEATURES]


class LGBM:
    """Gradient-boosted trees on all stores together: log of each normal day's sales against its store-month's
    average (in logs), from the calendar features; recent days weigh more (half-life one year)."""

    def __init__(self, panel: pd.DataFrame, origin: pd.Timestamp, half_life: float = 365.0, seed: int = 0):
        import lightgbm as lgb

        h = panel[panel.normal & (panel.date <= origin)].copy()
        h["y"] = np.log(h.sales) - np.log(h.sales).groupby([h.store, h.month]).transform("mean")
        age = (origin - h.date).dt.days.to_numpy()
        self.model = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=25,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                                       random_state=seed, verbose=-1)
        self.model.fit(lgb_features(h), h.y, sample_weight=0.5 ** (age / half_life),
                       categorical_feature=["store_code", "season_code", "special_code"])

    def weights(self, fp: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(lgb_features(fp))) * fp.open.to_numpy()


def split_cents(total: float, weights: np.ndarray) -> np.ndarray:
    """Integer cents in proportion to the weights, adding up exactly to the total (largest remainder)."""
    cents = int(round(total * 100))
    raw = weights / weights.sum() * cents
    base = np.floor(raw).astype(np.int64)
    base[np.argsort(-(raw - base), kind="stable")[:cents - int(base.sum())]] += 1
    return base


# ------------------------------------------------------------------------------------------
# Days-off holidays: their own level, whatever the weekday
# ------------------------------------------------------------------------------------------
DAYS_OFF = ["New Year's Day", "Easter", "Mother's Day", "Memorial Day", "Father's Day", "Independence Day",
            "Labor Day", "Thanksgiving Day", "Christmas Day"]
WINDOW = {"Christmas Day": (14, 0), "New Year's Day": (0, 14)}  # stay inside the holiday's own season


def _window(day: pd.Timestamp, name: str) -> pd.DatetimeIndex:
    b, a = WINDOW.get(name, (10, 10))
    return pd.date_range(day - pd.Timedelta(days=b), day + pd.Timedelta(days=a))


def holiday_ratios(panel: pd.DataFrame, origin: pd.Timestamp, shrink: float = 2.0) -> pd.DataFrame:
    """Each days-off holiday's sales against the average normal day around it (any weekday), per store, from
    occurrences before the origin, shrunk toward the average across stores (in logs)."""
    p = panel[(panel.date <= origin)].set_index(["store", "date"])
    ok = p.normal & (p.special == "")
    rows = []
    for (s, d), r in p[p.special.isin(DAYS_OFF) & p.normal].iterrows():
        win = [(s, x) for x in _window(d, r.special) if x != d]
        ref = p.sales.reindex(win)[ok.reindex(win).fillna(False).to_numpy()]
        if len(ref) >= 7:
            rows.append({"store": s, "special": r.special, "lr": np.log(r.sales / ref.mean())})
    e = pd.DataFrame(rows)
    pooled = e.groupby("special").lr.mean()
    g = e.groupby(["store", "special"]).lr.agg(["mean", "count"]).reset_index()
    g["lr"] = (g["mean"] * g["count"] + pooled.reindex(g.special).to_numpy() * shrink) / (g["count"] + shrink)
    allp = pd.MultiIndex.from_product([STORES, pooled.index], names=["store", "special"]).to_frame(index=False)
    allp = allp.merge(g[["store", "special", "lr"]], how="left", on=["store", "special"])
    allp["lr"] = allp.lr.fillna(allp.special.map(pooled))
    return allp.set_index(["store", "special"]).lr


def holiday_override(fp: pd.DataFrame, w: np.ndarray, ratios: pd.Series) -> np.ndarray:
    """Set each days-off holiday's weight to its usual ratio times the average weight of the normal days around it
    in the same forecast, so a holiday falling on a weekend does not also get the weekend's lift."""
    w = np.asarray(w, dtype=float).copy()
    ws = pd.Series(w, index=pd.MultiIndex.from_arrays([fp.store.to_numpy(), pd.DatetimeIndex(fp.date)]))
    plain = pd.Series((fp.special == "").to_numpy() & (fp.open.to_numpy() > 0), index=ws.index)
    for i in np.flatnonzero(fp.special.isin(DAYS_OFF).to_numpy() & (w > 0)):
        s, d, name = fp.store.iloc[i], pd.Timestamp(fp.date.iloc[i]), fp.special.iloc[i]
        win = [(s, x) for x in _window(d, name) if x != d]
        ref = ws.reindex(win)[plain.reindex(win).fillna(False).to_numpy()]
        if (s, name) in ratios.index and len(ref) >= 7:
            w[i] = np.exp(ratios[(s, name)]) * ref.mean()
    return w
