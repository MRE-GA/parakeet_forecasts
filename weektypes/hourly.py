"""Hourly patterns from Toast: how each store's day divides into hours, by type of week and day of the week, and its
average check by hour.

Days used: normal trading days (no closure or partial closure: daily/common.py) in the hourly extract (Oct 2024 to Oct
14, 2026). Each store has one pattern per day of the week, the same for every type of week: in the backtest
(weektypes/backtest_hourly.py) patterns by type of week split the days no better (hourly error 21.6% against
21.3%), so the types differ in how much each day sells (the run-rate), not in how a day divides into hours. Special
days (holidays and the days around them) are left out of those patterns and make patterns of their own from the same
day in earlier years (21.2% against 26.1% with the weekday's pattern).

Hours shown: the hours with at least 0.5% of the store's sales in the last six months; sales in other hours (a few
dollars before opening or after closing) join the nearest hour shown. An hour that opened later (Little Italy has
traded until 6pm since March 2026) takes its share only from the days since it opened; the other hours share the rest.

A pattern is the sales-weighted share of each hour in the day (sum over days of the hour's sales over the sum of the
days' sales), with recent days weighing more (RECENCY_HALF_LIFE, chosen in weektypes/backtest_hourly.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import HOURLY, LAST_DAY  # noqa: I001  (model puts daily/ on the path)
from common import SPECIAL, build_panel

RECENCY_HALF_LIFE = 180  # days: recent days weigh more (weektypes/backtest_hourly.py: best with 90-180 days)
CHECK_MONTHS = 12  # average check: the last 12 months, by hour and day of the week (weektypes/backtest_hourly.py)
RECENT_HOURS = pd.Timedelta(days=183)  # the hours a store trades now: its last six months
MIN_HOUR_SHARE = 0.005  # hours shown: those with at least this share of those six months' sales
OPENED_SHARE = 0.0025  # an hour counts as open in a month from this share of the month's sales


@dataclass
class Hours:
    shown: list[int]  # hours shown, in order
    opened: dict[int, pd.Timestamp]  # first day each shown hour traded (the extract's first day if always)

    def fold(self, hour: np.ndarray) -> np.ndarray:
        """Each hour to the nearest hour shown (ties: the earlier one)."""
        s = np.array(self.shown)
        return s[np.abs(hour[:, None] - s[None, :]).argmin(1)]


def load(last_day: pd.Timestamp = LAST_DAY) -> pd.DataFrame:
    """Hourly sales and checks with the day's calendar: weekday, week number, normal day, special day."""
    h = pd.read_csv(HOURLY, parse_dates=["date"])
    h = h[h.date <= last_day]
    panel = build_panel(str(last_day.date()))[["store", "date", "normal"]]
    h = h.merge(panel, on=["store", "date"], how="left")
    h["normal"] = h.normal.fillna(False).astype(bool)
    h["dow"] = h.date.dt.dayofweek
    h["week"] = h.date.dt.isocalendar().week.astype(int).clip(upper=52)
    h["special"] = SPECIAL.reindex(pd.DatetimeIndex(h.date)).fillna("").to_numpy()
    return h


def store_hours(h: pd.DataFrame, store: str, last_day: pd.Timestamp = LAST_DAY) -> Hours:
    g = h[h.store == store]
    recent = g[g.date > last_day - RECENT_HOURS].groupby("hour").net_sales.sum()
    shown = sorted(recent[recent / recent.sum() >= MIN_HOUR_SHARE].index)
    m = g.assign(m=g.date.dt.to_period("M")).groupby(["m", "hour"]).net_sales.sum().unstack().fillna(0)
    share = m.div(m.sum(1), axis=0)
    opened = {}
    for hr in shown:
        on = (share[hr] >= OPENED_SHARE).to_numpy() if hr in share else np.zeros(len(share), bool)
        # the first month from which the hour trades every month
        first = next(i for i in range(len(on)) if on[i:].all())
        opened[hr] = g.date.min() if first == 0 else share.index[first].start_time
    return Hours(shown, opened)


def weights(dates: pd.Series, asof: pd.Timestamp, half_life: float | None) -> np.ndarray:
    if not half_life:
        return np.ones(len(dates))
    return 0.5 ** ((asof - dates).dt.days.to_numpy() / half_life)


def folded(h: pd.DataFrame, store: str, hours: Hours) -> pd.DataFrame:
    """One store's rows with every hour moved to the nearest hour shown."""
    g = h[h.store == store].copy()
    g["hour"] = hours.fold(g.hour.to_numpy())
    keys = ["store", "date", "hour", "normal", "dow", "week", "special"]
    return g.groupby(keys, as_index=False)[["net_sales", "checks"]].sum()


def pattern(days: pd.DataFrame, hours: Hours, asof: pd.Timestamp, half_life: float | None,
            fallback: pd.DataFrame | None = None) -> np.ndarray:
    """Share of each hour shown in the day, from the given days (rows: store-day-hour, hours already folded).
    Hours that opened later take their share from the days since they opened (from the fallback days if none of
    the given days are that recent)."""
    def sums(d):
        w = weights(d.date, asof, half_life)
        by = pd.DataFrame({"hour": d.hour.to_numpy(), "s": d.net_sales.to_numpy() * w}).groupby("hour").s.sum()
        day = pd.DataFrame({"date": d.date.to_numpy(), "s": d.net_sales.to_numpy() * w}).groupby("date").s.sum()
        return by, day

    first = min(hours.opened.values())
    late = [hr for hr in hours.shown if hours.opened[hr] > first]
    out = np.zeros(len(hours.shown))
    p_late = {}
    for hr in late:
        src = days[days.date >= hours.opened[hr]]
        if src.date.nunique() < 4 and fallback is not None:
            src = fallback[fallback.date >= hours.opened[hr]]
        by, day = sums(src)
        p_late[hr] = by.get(hr, 0.0) / day.sum() if day.sum() > 0 else 0.0
    by, _ = sums(days[~days.hour.isin(late)])
    rest = 1 - sum(p_late.values())
    for i, hr in enumerate(hours.shown):
        out[i] = p_late[hr] if hr in p_late else rest * by.get(hr, 0.0) / by.sum()
    return out


def avg_check(days: pd.DataFrame, hours: Hours) -> np.ndarray:
    """Average check by hour shown (sales over checks)."""
    g = days.groupby("hour")[["net_sales", "checks"]].sum().reindex(hours.shown).fillna(0)
    tot = days.net_sales.sum() / max(days.checks.sum(), 1)
    return np.where(g.checks > 20, g.net_sales / g.checks.clip(lower=1), tot)
