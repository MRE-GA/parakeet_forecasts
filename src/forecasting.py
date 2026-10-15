"""Data preparation and the three forecasting models behind the Parakeet Cafe 2027 sales forecast.

Every model is fitted per store and the store forecasts are summed to the brand total, so new
openings never get mistaken for organic growth.

* Prophet       daily net sales; linear trend, multiplicative weekly + yearly seasonality and
                US holiday effects (Mother's Day, Memorial Day, Christmas Eve, ...).
* SARIMA        monthly trading-day-adjusted sales rate (log); seasonal ARIMA with drift,
                order chosen by AICc from a small candidate set.
* Holt-Winters  monthly trading-day-adjusted sales rate; exponential smoothing with damped
                additive trend and multiplicative seasonality.

"Trading-day adjusted" means monthly sales are divided by the sum of the store's day-of-week
weights over the days it trades that month, so the monthly models know that a month with five
Saturdays is worth more than one with four (Prophet sees this natively in the daily data).
"""
from __future__ import annotations

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

    def trading_days(self, months: pd.PeriodIndex, origin: pd.Timestamp) -> np.ndarray:
        """Weekday-weighted count of trading days in each month."""
        w = self.dow_weights(origin)
        days = pd.date_range(months[0].start_time, months[-1].end_time.normalize())
        weights = w[days.dayofweek] * self.open_flags(days)
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


def _months_after(origin: pd.Timestamp, last_month: pd.Period) -> pd.PeriodIndex:
    return pd.period_range(origin.to_period("M") + 1, last_month, freq="M")


# --------------------------------------------------------------------------------------------
# Models. Each returns (point forecast by month, sample paths by month x draw or None).
# --------------------------------------------------------------------------------------------
PROPHET_PARAMS = dict(
    growth="linear",
    seasonality_mode="multiplicative",
    yearly_seasonality=10,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
)


def prophet_forecast(sd: StoreData, origin: pd.Timestamp, last_month: pd.Period,
                     n_samples: int = 0, seed: int = 0, log_y: bool = False, **overrides):
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    m = Prophet(**{**PROPHET_PARAMS, **overrides}, holidays=HOLIDAYS, uncertainty_samples=n_samples)
    train = sd.train_days(origin)
    if log_y:
        train = train.assign(y=np.log(train.y))
    m.fit(train)
    months = _months_after(origin, last_month)
    future = pd.DataFrame({"ds": pd.date_range(months[0].start_time, months[-1].end_time.normalize())})
    open_ = sd.open_flags(future.ds)
    month_of_day = future.ds.dt.to_period("M").to_numpy()
    yhat = m.predict(future).yhat.to_numpy()
    daily = (np.exp(yhat) if log_y else yhat.clip(min=0)) * open_
    point = pd.Series(daily, index=month_of_day).groupby(level=0).sum().reindex(months)
    samples = None
    if n_samples:
        np.random.seed(seed)
        draws = m.predictive_samples(future)["yhat"]
        draws = (np.exp(draws) if log_y else np.clip(draws, 0, None)) * open_[:, None]
        samples = pd.DataFrame(draws, index=month_of_day).groupby(level=0).sum().reindex(months)
    return point, samples


SARIMA_CANDIDATES = [
    ((0, 0, 0), (0, 1, 0, 12)),  # seasonal naive with drift
    ((1, 0, 0), (0, 1, 0, 12)),
    ((1, 0, 0), (0, 1, 1, 12)),
    ((0, 0, 1), (0, 1, 1, 12)),
    ((1, 0, 1), (0, 1, 1, 12)),
    ((2, 0, 0), (0, 1, 1, 12)),
]


def sarima_forecast(sd: StoreData, origin: pd.Timestamp, last_month: pd.Period,
                    n_samples: int = 0, seed: int = 0):
    from statsmodels.tsa.arima.model import ARIMA

    hist = sd.monthly_clean(origin)
    y = np.log(hist.to_numpy() / sd.trading_days(hist.index, origin))
    best, best_spec = None, None
    for order, seasonal in SARIMA_CANDIDATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = ARIMA(y, order=order, seasonal_order=seasonal, trend="t").fit()
            except Exception:
                continue
        if np.isfinite(res.aicc) and (best is None or res.aicc < best.aicc):
            best, best_spec = res, (order, seasonal)
    months = _months_after(origin, last_month)
    td = sd.trading_days(months, origin)
    point = pd.Series(np.exp(best.forecast(len(months))) * td, index=months)
    samples = None
    if n_samples:
        sims = best.simulate(len(months), repetitions=n_samples, anchor="end",
                             rng=np.random.default_rng(seed))
        sims = np.asarray(sims).reshape(len(months), n_samples)
        samples = pd.DataFrame(np.exp(sims) * td[:, None], index=months)
    point.attrs["spec"] = f"SARIMA{best_spec[0]}{best_spec[1][:3]}12 + drift"
    return point, samples


def holt_winters_forecast(sd: StoreData, origin: pd.Timestamp, last_month: pd.Period,
                          n_samples: int = 0, seed: int = 0):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    hist = sd.monthly_clean(origin)
    rate = hist.to_numpy() / sd.trading_days(hist.index, origin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(
            rate, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
    months = _months_after(origin, last_month)
    td = sd.trading_days(months, origin)
    point = pd.Series(res.forecast(len(months)) * td, index=months)
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
