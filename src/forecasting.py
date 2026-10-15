"""Data preparation and the three forecasting models behind the Parakeet Cafe 2027 sales forecast.

Every model is fitted per store and the store forecasts are summed to the brand total, so new
openings never get mistaken for organic growth.

* Prophet       daily net sales; linear trend, weekly + yearly seasonality and US holiday
                effects (Mother's Day, Memorial Day, Christmas Eve, ...).
* SARIMA        monthly trading-day-adjusted sales rate; seasonal ARIMA, order chosen by an
                information criterion from a candidate set.
* Holt-Winters  monthly trading-day-adjusted sales rate; exponential smoothing with trend and
                seasonality.

Each model takes a config dict (hyperparameters + feature switches). The *_DEFAULT dicts are the
round-1 settings; src/tune.py searches alternatives and writes the adopted ones to
outputs/tuned_configs.json, which load_configs() picks up.

"Trading-day adjusted" means monthly sales are divided by the sum of the store's day-of-week
weights over the days it trades that month, so the monthly models know that a month with five
Saturdays is worth more than one with four (Prophet sees this natively in the daily data).
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parents[1]
DAILY_CSV = ROOT / "data" / "raw" / "parakeet_toast_daily.csv"
OUTPUTS = ROOT / "outputs"
TUNED_CONFIGS = OUTPUTS / "tuned_configs.json"

MODELS = ["Prophet", "SARIMA", "Holt-Winters"]

# First day of steady-state trading used for modelling. Toast history starts with the POS
# migration in Sep 2022; stores opened since then skip their 2-4 week opening ramp.
STORE_START = {
    "Carmel Valley": "2022-10-01",
    "La Jolla": "2022-10-01",
    "La Jolla Juicery": "2022-10-01",
    "Little Italy": "2022-10-01",
    "Coronado": "2022-10-01",
    "La Jolla Cove": "2022-11-01",  # opened 2022-09-30
    "Nautilus": "2022-11-01",  # opened 2022-09-28
    "Newport": "2023-08-15",  # opened 2023-07-27
    "Beverly Hills": "2023-10-01",  # soft-opened 2023-09-04
    "Brentwood": "2024-02-01",  # opened 2024-01-11
    "Carlsbad": "2024-03-15",  # opened 2024-03-08
}
STORES = list(STORE_START)
REGION = {
    "Carmel Valley": "San Diego", "La Jolla": "San Diego", "La Jolla Juicery": "San Diego",
    "Little Italy": "San Diego", "Coronado": "San Diego", "La Jolla Cove": "San Diego",
    "Nautilus": "San Diego", "Carlsbad": "San Diego", "Newport": "Orange County",
    "Beverly Hills": "Los Angeles", "Brentwood": "Los Angeles",
}
WEEKDAY_ONLY = {"Nautilus"}  # corporate-campus cafe, closed on weekends

# One-off disruptions: excluded from training (Prophet) or imputed (monthly models), but kept
# in the actuals every model is scored against.
ANOMALIES = [
    (None, "2023-08-20", "2023-08-20", "Tropical Storm Hilary closures"),
    ("Brentwood", "2025-01-07", "2025-01-25", "Los Angeles wildfires"),
    ("La Jolla", "2025-07-08", "2025-07-08", "unplanned closure"),
    ("La Jolla Juicery", "2025-07-08", "2025-07-08", "unplanned closure"),
    ("Little Italy", "2025-06-11", "2025-06-18", "closures"),
    ("Little Italy", "2025-09-29", "2025-09-30", "closures"),
]
# Any other trading day below this share of the store's rolling 29-day median is treated as a
# partial closure (power cut, private event, ...).
PARTIAL_CLOSURE_RATIO = 0.25


def load_daily() -> pd.DataFrame:
    return pd.read_csv(DAILY_CSV, parse_dates=["date"])


def holiday_table(years=range(2022, 2029)) -> pd.DataFrame:
    """US federal holidays plus the cafe-relevant days that move sales (Prophet format)."""
    years = list(years)
    rows = []
    for day, name in holidays.US(years=years).items():
        rows.append((pd.Timestamp(day), "Observed holiday" if "observed" in name else name))
    for y in years:
        may_sundays = pd.date_range(f"{y}-05-01", f"{y}-05-31", freq="W-SUN")
        june_sundays = pd.date_range(f"{y}-06-01", f"{y}-06-30", freq="W-SUN")
        nov_thursdays = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="W-THU")
        rows += [
            (pd.Timestamp(y, 2, 14), "Valentine's Day"),
            (pd.Timestamp(easter(y)), "Easter"),
            (may_sundays[1], "Mother's Day"),
            (june_sundays[2], "Father's Day"),
            (nov_thursdays[3] + pd.Timedelta(days=1), "Black Friday"),
            (pd.Timestamp(y, 12, 24), "Christmas Eve"),
            (pd.Timestamp(y, 12, 31), "New Year's Eve"),
        ]
    table = pd.DataFrame(rows, columns=["ds", "holiday"]).drop_duplicates()
    return table.sort_values(["ds", "holiday"]).reset_index(drop=True)


HOLIDAYS = holiday_table()


# --------------------------------------------------------------------------------------------
# Extra calendar features (switchable in the configs)
# --------------------------------------------------------------------------------------------
EVENT_COLUMNS = ["holiday_week", "spring_break", "thanksgiving_weekend"]


def event_features(ds: pd.Series) -> pd.DataFrame:
    """Multi-day periods that single-day holiday effects miss."""
    ds = pd.DatetimeIndex(ds)
    out = pd.DataFrame(index=range(len(ds)))
    # between Christmas and New Year's Eve (both have their own holiday effect)
    out["holiday_week"] = ((ds.month == 12) & (ds.day >= 26) & (ds.day <= 30)).astype(float)
    # school spring break: the two weeks around Easter (Easter itself is a holiday)
    spring = np.zeros(len(ds))
    for y in set(ds.year):
        e = pd.Timestamp(easter(y))
        spring[(ds >= e - pd.Timedelta(days=7)) & (ds <= e + pd.Timedelta(days=7)) & (ds != e)] = 1
    out["spring_break"] = spring
    # Saturday and Sunday after Thanksgiving
    tg = HOLIDAYS.loc[HOLIDAYS.holiday == "Thanksgiving Day", "ds"]
    weekend = set(tg + pd.Timedelta(days=2)) | set(tg + pd.Timedelta(days=3))
    out["thanksgiving_weekend"] = ds.isin(list(weekend)).astype(float)
    return out


def is_summer(ds: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(ds).month.isin([6, 7, 8])


@dataclass
class StoreData:
    """One store's daily history on a full calendar, with the trading calendar and cleaning flags."""

    store: str
    daily: pd.DataFrame  # index: date; columns: sales, open, anomaly, clean
    closed_holidays: frozenset

    @property
    def start(self) -> pd.Timestamp:
        return self.daily.index[0]

    @property
    def first_full_month(self) -> pd.Period:
        p = self.start.to_period("M")
        return p if self.start.day == 1 else p + 1

    def open_flags(self, dates) -> np.ndarray:
        """1.0 where the store is expected to trade (weekday rule + holidays it closes on)."""
        dates = pd.DatetimeIndex(dates)
        flags = np.ones(len(dates))
        if self.store in WEEKDAY_ONLY:
            flags[dates.dayofweek >= 5] = 0
        if self.closed_holidays:
            closed = HOLIDAYS.loc[HOLIDAYS.holiday.isin(self.closed_holidays), "ds"]
            flags[dates.isin(closed)] = 0
        return flags

    def train_days(self, origin: pd.Timestamp) -> pd.DataFrame:
        """Normal trading days up to the origin, as Prophet's (ds, y) frame."""
        d = self.daily.loc[: origin]
        d = d[(d.open == 1) & ~d.anomaly]
        return pd.DataFrame({"ds": d.index, "y": d.sales.to_numpy()})

    def monthly_clean(self, origin: pd.Timestamp) -> pd.Series:
        """Cleaned monthly sales for full months from the first full month to the origin month."""
        m = self.daily.loc[: origin, "clean"].groupby(lambda x: x.to_period("M")).sum()
        return m.loc[self.first_full_month:]

    def monthly_actual(self) -> pd.Series:
        return self.daily.sales.fillna(0).groupby(lambda x: x.to_period("M")).sum()

    def dow_weights(self, origin: pd.Timestamp) -> np.ndarray:
        """Relative sales by weekday over the last 52 weeks of training (mean of open days = 1)."""
        d = self.daily.loc[origin - pd.Timedelta(days=363): origin]
        d = d[d.open == 1]
        by_dow = d.clean.groupby(d.index.dayofweek).mean().reindex(range(7)).fillna(0)
        return (by_dow / d.clean.mean()).to_numpy()

    def trading_days(self, months: pd.PeriodIndex, origin: pd.Timestamp, mode: str = "weekday") -> np.ndarray:
        """Trading days per month: weekday-weighted ("weekday") or a plain count ("calendar")."""
        days = pd.date_range(months[0].start_time, months[-1].end_time.normalize())
        weights = self.open_flags(days)
        if mode == "weekday":
            weights = weights * self.dow_weights(origin)[days.dayofweek]
        per_month = pd.Series(weights, index=days.to_period("M")).groupby(level=0).sum()
        return per_month.reindex(months).to_numpy()


def prepare_store(raw: pd.DataFrame, store: str, last_date: str) -> StoreData:
    s = raw.loc[raw.store == store].set_index("date").net_sales
    idx = pd.date_range(STORE_START[store], last_date)
    d = pd.DataFrame({"sales": s.reindex(idx)}, index=idx)

    weekday_rule = np.where((store in WEEKDAY_ONLY) & (idx.dayofweek >= 5), 0, 1)
    # Holidays the store usually closes on (e.g. the Nautilus campus cafe on Thanksgiving).
    hol = HOLIDAYS[HOLIDAYS.ds.isin(idx[weekday_rule == 1])]
    traded = hol.ds.map(d.sales.gt(0))
    open_rate = traded.groupby(hol.holiday).mean()
    closed_holidays = frozenset(open_rate[open_rate < 0.5].index)
    sd = StoreData(store, d, closed_holidays)
    d["open"] = sd.open_flags(idx)

    anomaly = pd.Series(False, index=idx)
    for who, a, b, _ in ANOMALIES:
        if who in (None, store):
            anomaly.loc[a:b] = True
    trading = d.sales.where(d.open == 1)
    ratio = trading / trading.rolling(29, center=True, min_periods=10).median()
    anomaly |= (d.open == 1) & (d.sales.isna() | (ratio < PARTIAL_CLOSURE_RATIO))
    d["anomaly"] = anomaly & (d.open == 1)

    # Impute anomalous trading days with the median of the same weekday in the surrounding weeks.
    good = d.sales.where((d.open == 1) & ~d.anomaly)
    clean = good.copy()
    for day in d.index[d.anomaly]:
        for span in (28, 56, 112):
            window = good.loc[day - pd.Timedelta(days=span): day + pd.Timedelta(days=span)]
            same_dow = window[window.index.dayofweek == day.dayofweek].dropna()
            if len(same_dow) >= 2:
                clean.loc[day] = same_dow.median()
                break
    d["clean"] = clean.where(d.open == 1, 0).fillna(0)
    return sd


def load_stores(last_date: str = "2026-10-14") -> dict[str, StoreData]:
    raw = load_daily()
    return {s: prepare_store(raw, s, last_date) for s in STORES}


def aggregate_store(sds: list[StoreData], name: str = "All cafés") -> StoreData:
    """One series for a fixed set of stores (cleaned sales summed), from the date all are open.

    Used to test a single top-down model against the per-store (bottom-up) approach.
    """
    start = max(sd.start for sd in sds)
    end = min(sd.daily.index[-1] for sd in sds)
    idx = pd.date_range(start, end)
    clean = sum(sd.daily.clean.reindex(idx).fillna(0) for sd in sds)
    d = pd.DataFrame({"sales": clean, "open": 1.0, "anomaly": False, "clean": clean}, index=idx)
    return StoreData(name, d, frozenset())


def _months_after(origin: pd.Timestamp, last_month: pd.Period) -> pd.PeriodIndex:
    return pd.period_range(origin.to_period("M") + 1, last_month, freq="M")


# --------------------------------------------------------------------------------------------
# Models. Each returns (point forecast by month, sample paths by month x draw or None).
# --------------------------------------------------------------------------------------------
PROPHET_DEFAULT = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "seasonality_mode": "multiplicative",
    "changepoint_range": 0.8,
    "yearly_fourier": 10,
    "holiday_windows": False,  # extra regressors: holiday week, spring break, Thanksgiving weekend
    "summer_weekly": False,  # separate weekly pattern for Jun-Aug
    "cap_outliers": None,  # cap training days above k x the 29-day rolling median
    "log_y": False,
}


def prophet_forecast(sd: StoreData, origin: pd.Timestamp, last_month: pd.Period,
                     n_samples: int = 0, seed: int = 0, cfg: dict | None = None):
    from prophet import Prophet

    c = {**PROPHET_DEFAULT, **(cfg or {})}
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    m = Prophet(
        growth="linear",
        seasonality_mode=c["seasonality_mode"],
        yearly_seasonality=int(c["yearly_fourier"]),
        weekly_seasonality=not c["summer_weekly"],
        daily_seasonality=False,
        holidays=HOLIDAYS,
        changepoint_prior_scale=c["changepoint_prior_scale"],
        seasonality_prior_scale=c["seasonality_prior_scale"],
        holidays_prior_scale=c["holidays_prior_scale"],
        changepoint_range=c["changepoint_range"],
        uncertainty_samples=n_samples,
    )
    train = sd.train_days(origin)
    if c["cap_outliers"]:
        ceiling = train.y.rolling(29, center=True, min_periods=10).median() * c["cap_outliers"]
        train = train.assign(y=np.minimum(train.y, ceiling.fillna(np.inf)))
    if c["log_y"]:
        train = train.assign(y=np.log(train.y))

    months = _months_after(origin, last_month)
    future = pd.DataFrame({"ds": pd.date_range(months[0].start_time, months[-1].end_time.normalize())})

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        if c["summer_weekly"]:
            summer = is_summer(df.ds)
            df = df.assign(is_summer=summer, not_summer=~summer)
        if c["holiday_windows"]:
            df = pd.concat([df, event_features(df.ds)], axis=1)
        return df

    if c["summer_weekly"]:
        m.add_seasonality("weekly_summer", period=7, fourier_order=3, condition_name="is_summer")
        m.add_seasonality("weekly_rest", period=7, fourier_order=3, condition_name="not_summer")
    if c["holiday_windows"]:
        for col in EVENT_COLUMNS:
            m.add_regressor(col, prior_scale=c["holidays_prior_scale"])
    m.fit(add_features(train))
    future = add_features(future)

    open_ = sd.open_flags(future.ds)
    month_of_day = future.ds.dt.to_period("M").to_numpy()
    yhat = m.predict(future).yhat.to_numpy()
    daily = (np.exp(yhat) if c["log_y"] else yhat.clip(min=0)) * open_
    point = pd.Series(daily, index=month_of_day).groupby(level=0).sum().reindex(months)
    samples = None
    if n_samples:
        np.random.seed(seed)
        draws = m.predictive_samples(future)["yhat"]
        draws = (np.exp(draws) if c["log_y"] else np.clip(draws, 0, None)) * open_[:, None]
        samples = pd.DataFrame(draws, index=month_of_day).groupby(level=0).sum().reindex(months)
    return point, samples


SARIMA_DEFAULT = {
    "trend": "drift",  # drift: seasonal differencing + linear trend (constant YoY growth)
    #                    none: seasonal differencing, no growth term; local: also first-differenced
    "candidates": "small",  # small: 6 orders; extended: p<=2, q<=1, Q<=1
    "criterion": "aicc",  # or "bic"
    "transform": "log",  # or "none"
    "td_adjust": "weekday",  # or "calendar"
}
_SARIMA_SMALL = [((0, 0), 0), ((1, 0), 0), ((1, 0), 1), ((0, 1), 1), ((1, 1), 1), ((2, 0), 1)]  # ((p,q),Q)
_SARIMA_EXTENDED = [((p, q), Q) for p in (0, 1, 2) for q in (0, 1) for Q in (0, 1)]


def sarima_forecast(sd: StoreData, origin: pd.Timestamp, last_month: pd.Period,
                    n_samples: int = 0, seed: int = 0, cfg: dict | None = None):
    from statsmodels.tsa.arima.model import ARIMA

    c = {**SARIMA_DEFAULT, **(cfg or {})}
    hist = sd.monthly_clean(origin)
    rate = hist.to_numpy() / sd.trading_days(hist.index, origin, c["td_adjust"])
    y = np.log(rate) if c["transform"] == "log" else rate
    d, trend = {"drift": (0, "t"), "none": (0, "n"), "local": (1, "n")}[c["trend"]]
    candidates = _SARIMA_SMALL if c["candidates"] == "small" else _SARIMA_EXTENDED
    best, best_spec, best_score = None, None, np.inf
    for (p, q), Q in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = ARIMA(y, order=(p, d, q), seasonal_order=(0, 1, Q, 12), trend=trend).fit()
            except Exception:
                continue
        score = res.aicc if c["criterion"] == "aicc" else res.bic
        if np.isfinite(score) and score < best_score:
            best, best_spec, best_score = res, ((p, d, q), (0, 1, Q)), score
    months = _months_after(origin, last_month)
    td = sd.trading_days(months, origin, c["td_adjust"])
    inverse = np.exp if c["transform"] == "log" else (lambda v: np.clip(v, 0, None))
    point = pd.Series(inverse(best.forecast(len(months))) * td, index=months)
    samples = None
    if n_samples:
        sims = best.simulate(len(months), repetitions=n_samples, anchor="end",
                             rng=np.random.default_rng(seed))
        sims = np.asarray(sims).reshape(len(months), n_samples)
        samples = pd.DataFrame(inverse(sims) * td[:, None], index=months)
    drift = " + drift" if trend == "t" else ""
    point.attrs["spec"] = f"SARIMA{best_spec[0]}{best_spec[1]}12{drift}"
    return point, samples


HW_DEFAULT = {
    "trend": "add",  # or None
    "damped": True,
    "phi": None,  # damping factor; None = estimated
    "seasonal": "mul",  # or "add"
    "boxcox": False,  # or "log"
    "init": "estimated",  # or "heuristic"
    "td_adjust": "weekday",  # or "calendar"
}


def holt_winters_forecast(sd: StoreData, origin: pd.Timestamp, last_month: pd.Period,
                          n_samples: int = 0, seed: int = 0, cfg: dict | None = None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    c = {**HW_DEFAULT, **(cfg or {})}
    hist = sd.monthly_clean(origin)
    rate = hist.to_numpy() / sd.trading_days(hist.index, origin, c["td_adjust"])
    damped = bool(c["trend"]) and c["damped"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(
            rate, trend=c["trend"], damped_trend=damped, seasonal=c["seasonal"], seasonal_periods=12,
            initialization_method=c["init"], use_boxcox=c["boxcox"],
        ).fit(optimized=True, damping_trend=c["phi"] if damped else None)
    months = _months_after(origin, last_month)
    td = sd.trading_days(months, origin, c["td_adjust"])
    point = pd.Series(np.clip(res.forecast(len(months)), 0, None) * td, index=months)
    samples = None
    if n_samples:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sims = res.simulate(len(months), repetitions=n_samples, error="mul", anchor="end",
                                rng=np.random.default_rng(seed))
        samples = pd.DataFrame(np.asarray(sims).reshape(len(months), n_samples) * td[:, None],
                               index=months)
    return point, samples


FORECASTERS = {
    "Prophet": prophet_forecast,
    "SARIMA": sarima_forecast,
    "Holt-Winters": holt_winters_forecast,
}
DEFAULT_CONFIGS = {"Prophet": PROPHET_DEFAULT, "SARIMA": SARIMA_DEFAULT, "Holt-Winters": HW_DEFAULT}


def load_configs() -> dict:
    """Adopted settings from src/tune.py if present, else the round-1 defaults.

    Returns {"models": {model: cfg}, "blend": "mean" | "median" | weights dict}.
    """
    if TUNED_CONFIGS.exists():
        adopted = json.loads(TUNED_CONFIGS.read_text())["adopted"]
        return {"models": {m: {**DEFAULT_CONFIGS[m], **adopted["models"][m]} for m in MODELS},
                "blend": adopted["blend"]}
    return {"models": {m: dict(DEFAULT_CONFIGS[m]) for m in MODELS}, "blend": "mean"}


def combine(frame: pd.DataFrame, rule) -> pd.Series:
    """Blend the three model columns: simple mean, median, or fixed weights."""
    if rule == "mean":
        return frame[MODELS].mean(axis=1)
    if rule == "median":
        return frame[MODELS].median(axis=1)
    w = pd.Series(rule)[MODELS]
    return frame[MODELS].mul(w / w.sum(), axis=1).sum(axis=1)


# --------------------------------------------------------------------------------------------
# Accuracy metrics
# --------------------------------------------------------------------------------------------
def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: sum |F - A| / sum A."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return np.abs(forecast - actual).sum() / actual.sum()


def mape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return np.mean(np.abs(forecast - actual) / actual)


def bias(actual, forecast) -> float:
    """Signed total error: positive = over-forecast."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return (forecast.sum() - actual.sum()) / actual.sum()
