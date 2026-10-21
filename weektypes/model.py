"""Week types for payroll: each store's next 52 weeks grouped into 2 to 4 types of week, so that payroll can schedule
each type from one template (weektypes/templates.py: the type's average week by day and hour).

Week types. The weeks are the next 52 Monday-to-Sunday weeks (from Monday Oct 26, 2026, the week after this was
built, to Sunday Oct 24, 2027) and their sales are today's run-rate by day (report/build_daily_targets.py:
outputs/runrate_2026_daily.csv to Dec 31, 2026, then outputs/targets_2027_daily.csv). Each store's weeks are split into
types in runs of at least four weeks in a row, the first and the last run included, so a type is "from week X to
week Y", never one week here and one there. The split minimises the squared log difference between each week and its
type's level, plus a cost for every change of type (SWITCH), so two close types do not alternate through the year; the
store gets as many types as it can (at most four) while every type's average week (as its template has it: holiday
weeks with a template of their own left out) sells at least 10% more than the one below it, and never fewer than two.
Types are named by level: Low and High; Low, Normal and High; or Low, Normal, High and Super high.

Holiday weeks. A week with a special day (the daily model's list: daily/common.py) or the week after Easter (spring
break) does not pull the types: it counts a quarter in the split, so it takes the type it is closest to, and nothing
in the type's level. It becomes a holiday week, with a template of its own, when its run-rate sells at least 10% more
or less than its type's average week (the same threshold as between types). The other holiday weeks are ordinary
weeks of their type; their special days are listed on the type's page.

La Jolla Cove. The run-rate carries the budget's remodeled café from its October 1, 2027 reopening ($125K a month), so
its types are found on the weeks before the reopening; the week of the reopening is a holiday week of its own and the
three weeks after it are one more type, Remodeled.

    python weektypes/model.py   -> each store's types, their weeks and holiday weeks (printed)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "daily"))
from common import EASTER, NAME, SPECIAL, STORES  # noqa: E402

OUT = ROOT / "outputs"
HOURLY = ROOT / "data" / "raw" / "parakeet_toast_hourly.csv.gz"
RUNRATE_27 = OUT / "targets_2027_daily.csv"
RUNRATE_26 = OUT / "runrate_2026_daily.csv"
WEEK1 = pd.Timestamp("2026-10-26")  # the next 52 weeks: from the Monday after this was built (Oct 21, 2026)
N_WEEKS = 52
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_RUN = 4  # weeks in a row, at least
CIRCULAR = False  # the next 52 weeks are a sequence, not a calendar year that repeats: the edges need 4 weeks too
SWITCH = 4 * 0.05 ** 2  # a change of type costs as much as four weeks 5% off their type's level (in logs)
HOLIDAY_WEIGHT = 0.25  # holiday weeks lean to the type they are closest to without moving the ranges much
STEP = 0.10  # a type sells at least this much more than the type below it; also the holiday-week threshold
MAX_TYPES, MIN_TYPES = 4, 2
TYPE_NAMES = {2: ["Low", "High"], 3: ["Low", "Normal", "High"], 4: ["Low", "Normal", "High", "Super high"]}
REMODELED = "Remodeled"
COVE = "La Jolla Cove"
COVE_REOPENS = pd.Timestamp("2027-10-01")
HOURS_FROM = pd.Timestamp("2024-10-01")  # the hourly extract
LAST_DAY = pd.Timestamp("2026-10-14")  # the last day of Toast data
DISPLAY = {s: NAME.get(s, s) for s in STORES}


def week_start(w: int) -> pd.Timestamp:
    return WEEK1 + pd.Timedelta(weeks=w - 1)


def week_of(d: pd.Timestamp) -> int:
    return (d - WEEK1).days // 7 + 1


# ------------------------------------------------------------------------------------------
# The run-rate by day and week
# ------------------------------------------------------------------------------------------
def window() -> pd.DatetimeIndex:
    return pd.date_range(WEEK1, week_start(N_WEEKS) + pd.Timedelta(days=6))


def runrate_days() -> pd.DataFrame:
    """Today's run-rate by day (rows) and store (Toast name) over the 52 weeks: the rest of 2026, then 2027."""
    back = {v: k for k, v in NAME.items()}
    r = pd.concat([pd.read_csv(RUNRATE_26, parse_dates=["date"]), pd.read_csv(RUNRATE_27, parse_dates=["date"])])
    r = r.assign(store=r.store.map(lambda s: back.get(s, s))).set_index(["store", "date"]).runrate
    out = r.unstack(0).reindex(window())[STORES]
    assert not out.isna().any().any(), "the run-rate does not cover the 52 weeks"
    return out


# ------------------------------------------------------------------------------------------
# Holiday weeks
# ------------------------------------------------------------------------------------------
def holiday_weeks() -> dict[int, list[tuple[pd.Timestamp, str]]]:
    """Weeks with a special day (the daily model's list), and the week after Easter (spring break)."""
    days = window()
    sp = SPECIAL.reindex(days).dropna()
    out: dict[int, list] = {}
    for d, lab in sp.items():
        if lab == "Observed holiday":  # named after the holiday it stands for (the day before or after)
            near = [SPECIAL.get(d + pd.Timedelta(days=k)) for k in (-1, 1)]
            lab = next(f"{n} (observed)" for n in near if n and n != "Observed holiday")
        out.setdefault(week_of(d), []).append((d, lab))
    for e in EASTER.values():
        if e + pd.Timedelta(days=7) in days:
            out.setdefault(week_of(e + pd.Timedelta(days=1)), []).append((e + pd.Timedelta(days=1), "Spring break"))
    return dict(sorted(out.items()))


HOLIDAY_LABEL = {  # how a holiday week is called on its page
    "Easter": "Easter week", "Spring break": "Spring break", "Mother's Day": "Mother's Day week",
    "Memorial Day": "Memorial Day week", "Father's Day": "Father's Day week",
    "Independence Day": "Fourth of July week", "Independence Day (observed)": "Week after the Fourth",
    "Juneteenth National Independence Day": "Juneteenth week",
    "Labor Day": "Labor Day week", "Thanksgiving Day": "Thanksgiving week", "Christmas Day": "Christmas week",
    "New Year's Day": "New Year's week", "Martin Luther King Jr. Day": "MLK Day week",
    "Washington's Birthday": "Presidents' Day week", "Valentine's Day": "Valentine's week",
    "Columbus Day": "Columbus Day week", "Veterans Day": "Veterans Day week",
}


def holiday_week_name(days: list[tuple[pd.Timestamp, str]]) -> str:
    """A holiday week is named after its most important day."""
    order = list(HOLIDAY_LABEL)
    labs = sorted((lab for _, lab in days if lab in HOLIDAY_LABEL), key=order.index)
    for key in ["Thanksgiving Day", "Christmas Day", "New Year's Day", "Independence Day", "Easter", "Mother's Day",
                "Memorial Day", "Labor Day"]:
        if key in labs:
            return HOLIDAY_LABEL[key]
    return HOLIDAY_LABEL[labs[0]] if labs else days[0][1]


# ------------------------------------------------------------------------------------------
# Splitting the year into types
# ------------------------------------------------------------------------------------------
def _dp(x: np.ndarray, wt: np.ndarray, mu: np.ndarray, min_run: int, penalty: float) -> tuple[float, np.ndarray]:
    """Best labels for a sequence (no wrap): every run of one label at least min_run long, each new run costing the
    penalty. State: (label, run length capped at min_run)."""
    n, k, R = len(x), len(mu), min_run
    cost = wt[:, None] * (x[:, None] - mu[None, :]) ** 2
    D = np.full((n, k, R), np.inf)
    came = np.zeros((n, k), dtype=np.int64)  # new run (c = 0): the label before it
    stayed = np.zeros((n, k), dtype=bool)  # c = R - 1: came from R - 1 (else from R - 2)
    D[0, :, 0] = cost[0]
    ar = np.arange(k)
    for i in range(1, n):
        prev = D[i - 1]
        D[i, :, 1:] = prev[:, :-1] + cost[i][:, None]
        stay = prev[:, R - 1] + cost[i]
        stayed[i] = stay < D[i, :, R - 1]
        D[i, stayed[i], R - 1] = stay[stayed[i]]
        fin = prev[:, R - 1]
        o = np.argsort(fin)
        other = np.where(ar == o[0], o[1], o[0]) if k > 1 else ar
        D[i, :, 0] = fin[other] + penalty + cost[i]
        came[i] = other
    l = int(np.argmin(D[n - 1, :, R - 1]))
    total = float(D[n - 1, l, R - 1])
    lab = np.zeros(n, dtype=int)
    c = R - 1
    for i in range(n - 1, -1, -1):
        lab[i] = l
        if c == 0:
            l, c = int(came[i, l]), R - 1
        elif c == R - 1 and stayed[i, l]:
            c = R - 1
        else:
            c -= 1
    return total, lab


def _assign(x, wt, mu, circular) -> tuple[float, np.ndarray]:
    if not circular:
        return _dp(x, wt, mu, MIN_RUN, SWITCH)
    best = (np.inf, None)
    for r in range(len(x)):  # every starting week; the best one starts at a boundary, so runs wrap correctly
        cost, lab = _dp(np.roll(x, -r), np.roll(wt, -r), mu, MIN_RUN, SWITCH)
        cost += SWITCH * (lab[0] != lab[-1])  # December into January
        if cost < best[0]:
            best = (cost, np.roll(lab, r))
    return best


def segment(x: np.ndarray, holiday: np.ndarray, k: int, circular: bool = True):
    """k levels and runs of at least MIN_RUN weeks minimising each week's squared distance to its level (holiday weeks
    at a quarter weight), plus SWITCH for every change of type; Lloyd iterations from several starts, levels from the
    weeks that are not holiday weeks. Returns (cost, labels ranked by level, levels)."""
    wt = np.where(holiday, HOLIDAY_WEIGHT, 1.0)
    fixed = x[~holiday]
    starts = [np.quantile(fixed, (np.arange(k) + 0.5) / k)]
    rng = np.random.default_rng(0)
    starts += [np.sort(rng.choice(fixed, k, replace=False)) for _ in range(12)]
    best = (np.inf, None, None)
    for mu in starts:
        mu = np.array(mu, dtype=float)
        lab, cost = None, np.inf
        for _ in range(50):
            cost, new = _assign(x, wt, mu, circular)
            if lab is not None and (new == lab).all():
                break
            lab = new
            if len(set(lab[~holiday])) < k:
                break
            mu = np.array([x[(lab == l) & ~holiday].mean() for l in range(k)])
        if len(set(lab[~holiday])) < k:
            continue
        if cost < best[0] - 1e-12:
            order = np.argsort(mu)
            rank = np.empty(k, dtype=int)
            rank[order] = np.arange(k)
            best = (cost, rank[lab], mu[order])
    return best


@dataclass
class StoreWeeks:
    store: str  # Toast name
    weekly: pd.Series  # run-rate by week (1..52)
    types: list[str]  # type names, low to high
    label: pd.Series  # type of every week (holiday weeks: the type around them)
    holiday: dict[int, str] = field(default_factory=dict)  # holiday weeks with a template of their own: week -> name
    holiday_days: dict[int, list] = field(default_factory=dict)  # every holiday week's special days
    tried: dict[int, list[float]] = field(default_factory=dict)  # levels found for each number of types

    @property
    def name(self) -> str:
        return DISPLAY[self.store]

    def regular(self, t: str) -> list[int]:
        return [w for w in self.label.index if self.label[w] == t and w not in self.holiday]

    def avg_week(self, t: str) -> float:
        return float(self.weekly[self.regular(t)].mean())

    def ranges(self, t: str) -> list[tuple[int, int]]:
        """The runs of a type (holiday weeks included in the run they sit in). A run over the end of the 52 weeks into
        their start is shown as two: they are a year apart."""
        on = (self.label == t).to_numpy()
        runs, i, n = [], 0, len(on)
        while i < n:
            if on[i]:
                j = i
                while j + 1 < n and on[j + 1]:
                    j += 1
                runs.append([i + 1, j + 1])
                i = j + 1
            else:
                i += 1
        return [tuple(r) for r in runs]


def holiday_own(label: pd.Series, weekly: pd.Series, hol: dict, fixed: set) -> set:
    """The holiday weeks that get a template of their own: those that sell 10% more or less than their type's average
    week, the average without them (start from all of them apart and settle). Weeks in fixed always do."""
    own = set(hol) | fixed
    for _ in range(20):
        base = {t: weekly[[w for w in label.index if label[w] == t and w not in own]].mean() for t in set(label)}
        new = {w for w in hol if abs(weekly[w] / base[label[w]] - 1) >= STEP} | fixed
        if new == own:
            break
        own = new
    return own


def classify_store(store: str, weekly: pd.Series, hol: dict) -> StoreWeeks:
    weeks = weekly.index.to_numpy()
    span, fixed, after = weeks, set(), None
    if store == COVE:  # types on the weeks before the reopening; the reopening week and the weeks after apart
        reopen = week_of(COVE_REOPENS)
        span, fixed = weeks[weeks < reopen], {reopen}
        after = pd.Series(REMODELED, index=weeks[weeks >= reopen])
    x = np.log(weekly[span].to_numpy())
    in_hol = np.isin(span, list(hol))
    tried, fits = {}, {}
    for k in range(MIN_TYPES, MAX_TYPES + 1):
        cost, lab, mu = segment(x, in_hol, k, circular=CIRCULAR)
        if lab is None:
            continue
        label = pd.Series(lab, index=span).map(dict(enumerate(TYPE_NAMES[k])))
        if after is not None:
            label = pd.concat([label, after])
        own = holiday_own(label, weekly, hol, fixed)
        # the type's average week as its template has it: holiday weeks with their own template left out
        tried[k] = [weekly[[w for w in span if label[w] == t and w not in own]].mean() for t in TYPE_NAMES[k]]
        fits[k] = (label, own)
    # as many types as possible while each sells at least 10% more than the one below; never fewer than two
    ok = [k for k, avg in tried.items() if all(hi / lo >= 1 + STEP for lo, hi in zip(avg, avg[1:]))]
    k = max(ok, default=MIN_TYPES)
    label, own = fits[k]
    types = TYPE_NAMES[k] + ([REMODELED] if after is not None else [])
    sw = StoreWeeks(store, weekly, types, label.sort_index(), tried=tried,
                    holiday_days={w: d for w, d in hol.items()})
    sw.holiday = {w: ("Reopening week" if w in fixed else holiday_week_name(hol[w])) for w in sorted(own)}
    return sw


def classify(rr: pd.DataFrame | None = None) -> dict[str, StoreWeeks]:
    rr = runrate_days() if rr is None else rr
    weekly = rr.groupby((rr.index - WEEK1).days // 7 + 1).sum()
    hol = holiday_weeks()
    return {s: classify_store(s, weekly[s], hol) for s in STORES}


if __name__ == "__main__":
    res = classify()
    for s, sw in res.items():
        parts = []
        for t in sw.types:
            rng = ", ".join(f"{a}-{b}" for a, b in sw.ranges(t))
            parts.append(f"{t} [{rng}] ${sw.avg_week(t)/1000:.1f}K")
        hol = ", ".join(f"W{w} {n} ({sw.weekly[w]/sw.avg_week(sw.label[w])-1:+.0%})" for w, n in sw.holiday.items())
        lv = {k: "/".join(f"{v/1000:.1f}" for v in a) for k, a in sw.tried.items()}
        print(f"{sw.name}: {' | '.join(parts)}\n   holiday weeks: {hol}\n   levels tried: {lv}")
