"""Hourly patterns from Toast: how each store's day divides into hours, by type of week and day of the week, and its
average check by hour.

Days used: normal trading days (no closure or partial closure: daily/common.py) in the hourly extract (Oct 2024 to Oct
14, 2026); a past day takes the type of its week of the 52 (model.window_week).

The pattern for a weekday and type of week is the store's pattern for that weekday, recent days weighing more
(half-life RECENCY_HALF_LIFE days), times the type's seasonal shift of the hours (in summer some stores sell more of
the day in the afternoon, others in the morning), and normalised. The shift is each type's share of every hour against
the share over all the days of the same 52-week year, all days of the week together, averaged over the full years in
the data, so that the drift from one year to the next (each year more of the day is sold early) is not read as a
season; the weekday patterns are learned on days with their type's shift taken out, so the recent months (summer and
early fall) do not give their hours to the winter types. Half the shift is used (SEASON_SHIFT): it repeats from one
year to the next in direction at most stores but not in size, and summer 2026 started earlier in the day than summer
2025. Backtests (weektypes/backtest_hourly.py; hourly error splitting the actual day): the shift, in full or half,
weekdays and weekends apart or not, against one pattern per weekday, and a pattern of its own per type and weekday
(21.6% against 21.3%: too few days each); recency, with the half shift and a 90-day half-life best (20.66% against
20.71% for the weekday pattern alone with 180 days). Special days (holidays and the days around them) are left out of
those patterns and make patterns of their own from the same day in earlier years (24.9% against 26.0%).

Hours shown: the hours with at least 0.5% of the store's sales in the last six months; sales in other hours (a few
dollars before opening or after closing) join the nearest hour shown. An hour that opened later (Little Italy has
traded until 6pm, Friday to Sunday, since March 2026) takes its share only from the days since it opened, and no
seasonal shift; the other hours share the rest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import HOURLY, LAST_DAY, WEEK1, window_week  # noqa: I001  (model puts daily/ on the path)
from common import SPECIAL, build_panel

RECENCY_HALF_LIFE = 90  # days: recent days weigh more (weektypes/backtest_hourly.py)
SEASON_SHIFT = 0.5  # how much of each type's seasonal shift of the hours is used (weektypes/backtest_hourly.py)
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
    h["week"] = window_week(h.date)  # the week of the 52 it matches, for its type
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


DAYPARTS = [(0, 10), (11, 13), (14, 23)]  # open to 10am, 11am to 1pm, 2pm to close


def season_shift(days: pd.DataFrame, hours: Hours, by: list[str], dayparts: bool = False,
                 min_weeks: int = 48) -> dict[tuple, np.ndarray]:
    """How each type of week (or other group of days) moves the day's hours: its pattern over all its days, every day
    of the week together and without recency weights, against the pattern of all the days of the same 52-week year,
    hour by hour (or by part of the day), averaged over the full 52-week years in the data (at least min_weeks of
    their weeks; if none, all the days at once). Measuring it within each year keeps the drift from year to year (more
    sales early in the day each year) out of the seasons. Pooling the days of the week keeps each type's shift on many
    days; hours that opened later get none."""
    first = min(hours.opened.values())
    steady = [hr for hr in hours.shown if hours.opened[hr] == first]
    groups = ([[hr for hr in steady if lo <= hr <= hi] for lo, hi in DAYPARTS] if dayparts else [[hr] for hr in steady])
    groups = [gr for gr in groups if gr]

    def share(d):
        t = d[d.hour.isin(steady)].groupby("hour").net_sales.sum().reindex(steady).fillna(0)
        return np.array([t[gr].sum() for gr in groups]) / t.sum()

    def ratios(d):
        p_all = share(d)
        out = {}
        for key, g in d.groupby(by):
            out[key if isinstance(key, tuple) else (key,)] = (
                np.divide(share(g), p_all, out=np.ones(len(groups)), where=p_all > 0), g.date.nunique())
        return out

    year = (days.date - WEEK1).dt.days // 364
    full = [y for y, d in days.groupby(year) if d.week.nunique() >= min_weeks]
    per_year = [ratios(days[year == y]) for y in full] or [ratios(days)]
    out = {}
    for key in set().union(*per_year):
        got = [r[key] for r in per_year if key in r]
        avg = sum(m * n for m, n in got) / sum(n for _, n in got)
        m = np.ones(len(hours.shown))
        for gr, v in zip(groups, avg):
            for hr in gr:
                m[hours.shown.index(hr)] = v
        out[key] = m
    return out


@dataclass
class Patterns:
    """Hourly patterns: one per day of the week with the seasons taken out, times each type's shift."""
    base: np.ndarray  # 7 x hours
    shift: dict[tuple, np.ndarray]  # key -> multiplier by hour (empty: no shift)
    by: tuple[str, ...]

    def get(self, dow: int, key: tuple = ()) -> np.ndarray:
        p = self.base[dow] * self.shift.get(key, 1.0)
        return p / p.sum()


def fit_patterns(days: pd.DataFrame, hours: Hours, asof: pd.Timestamp, half_life: float | None,
                 by: tuple[str, ...] = ("type",), strength: float = 1.0, dayparts: bool = False) -> Patterns:
    """The weekday patterns (recent days weighing more) of days with their type's shift taken out, and the shifts
    (raised to strength: 0 is none, 1 the full shift)."""
    shift = ({k: m ** strength for k, m in season_shift(days, hours, list(by), dayparts).items()}
             if strength else {})
    adj = days
    if shift:
        idx = {hr: i for i, hr in enumerate(hours.shown)}
        keys = zip(*(days[c] for c in by))
        mult = np.array([shift[k][idx[hr]] for k, hr in zip(keys, days.hour)])
        adj = days.assign(net_sales=days.net_sales / mult)
    base = np.array([pattern(adj[adj.dow == d], hours, asof, half_life) for d in range(7)])
    return Patterns(base, shift, by)


def avg_check(days: pd.DataFrame, hours: Hours) -> np.ndarray:
    """Average check by hour shown (sales over checks)."""
    g = days.groupby("hour")[["net_sales", "checks"]].sum().reindex(hours.shown).fillna(0)
    tot = days.net_sales.sum() / max(days.checks.sum(), 1)
    return np.where(g.checks > 20, g.net_sales / g.checks.clip(lower=1), tot)
