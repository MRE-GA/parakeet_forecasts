"""Does splitting each store's sales by channel (in the café, with online pickup, against delivery) divide a month
into its days better?

Delivery has its own week (weekday lunches, little weekend lift) and its share grew every year, so a model of the
total learns a blend of two weeks whose weights keep moving. Here each channel gets the chosen daily model (Prophet
with the extended special days x the pooled tree model), and a store-month's days are the two channel shapes weighted
by the month's channel mix: the mix of the last three (or twelve) months before the origin, or, as a ceiling, the
month's actual mix. Two groupings: online pickup with the café (as asked), or with delivery (all online orders).
Same backtest and metric as daily/backtest.py (daily error of the total, given the true month).

    python daily/channels.py   (after daily/backtest.py ... daily/prophet_variants.py)
        -> daily/outputs/channels_mix.csv, channels_weekday.csv, channels_backtest.csv, channels_by_store.csv,
           channels.png
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DAYS_OFF, LAST_DAY, LGBM, NAME, OUT, STORES, build_panel, shape_errors  # noqa: E402
from forecasting import PARTIAL_CLOSURE_RATIO, load_stores  # noqa: E402
from prophet_variants import fit as prophet_fit  # noqa: E402

CHANNEL_CSV = HERE / "data" / "toast_channels_daily.csv"
COLS = {"cafe": ["in_store", "pickup"], "delivery": ["delivery"], "in_store": ["in_store"],
        "online": ["pickup", "delivery"]}
GROUPINGS = {"ch": ("cafe", "delivery"), "on": ("in_store", "online")}  # (always modelled, modelled where big)
MIN_SHARE = 0.03  # below this share of the last 12 months the second channel stays in the first (La Jolla Cove)
FINAL_SHRINK = {"Beverly Hills": 0.5}  # as in report/build_daily_targets.py
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def channel_panel() -> pd.DataFrame:
    p = build_panel()
    ch = pd.read_csv(CHANNEL_CSV, parse_dates=["date"])
    p = p.merge(ch, on=["store", "date"], how="left")
    for c in ("in_store", "pickup", "delivery"):
        p[c] = p[c].fillna(0)
    p["cafe"], p["online"] = p.in_store + p.pickup, p.pickup + p.delivery
    return p


def marginal(p: pd.DataFrame, col: str) -> pd.Series:
    """Store-months when a channel was still marginal (under 2% of sales): too thin to learn its days from."""
    g = p.groupby(["store", "month"])[[col, "sales"]].transform("sum")
    return g[col] < 0.02 * g.sales


def channel_store(sd, hist: pd.DataFrame, channel: str):
    """The store's data with one channel's sales; days that channel did not trade normally (no sales, under a
    quarter of its usual level, e.g. a delivery app down, or the channel still marginal) are left out of training."""
    d = sd.daily.copy()
    s = hist.set_index("date")[channel].reindex(d.index)
    d["sales"] = s
    trading = s.where(d.open == 1)
    ratio = trading / trading.rolling(29, center=True, min_periods=10).median()
    thin = marginal(hist, channel).set_axis(hist.date).reindex(d.index, fill_value=True).to_numpy()
    d["anomaly"] = d.anomaly | ((d.open == 1) & ~(s > 0)) | (ratio < PARTIAL_CLOSURE_RATIO) | thin
    return replace(sd, daily=d)


def channel_tree_panel(panel: pd.DataFrame, channel: str) -> pd.DataFrame:
    p = panel.copy()
    p["sales"] = p[channel]
    ratio = p.sales / p.groupby("store").sales.transform(lambda x: x.rolling(29, center=True, min_periods=10).median())
    p["normal"] = p.normal & (p.sales > 0) & (ratio >= PARTIAL_CLOSURE_RATIO) & ~marginal(panel, channel)
    return p


def mix_last(panel: pd.DataFrame, origin: pd.Timestamp, months: int, col: str) -> pd.Series:
    """A channel's share of each store's sales over the last full months before the origin."""
    h = panel[(panel.month > origin.to_period("M") - months) & (panel.month <= origin.to_period("M"))]
    g = h.groupby("store")[[col, "sales"]].sum()
    return g[col] / g.sales


def combine(test: pd.DataFrame, first: str, second: str, share: pd.Series) -> np.ndarray:
    """Each store-month's days: the two channel shapes (each adding up to 1 over the month) weighted by the second
    channel's share."""
    key = [test.origin, test.store, test.month]
    a = test[first] / test.groupby(key)[first].transform("sum")
    b = (test[second] / test.groupby(key)[second].transform("sum")).fillna(0)
    return ((1 - share) * a + share * b).to_numpy()


def describe(panel: pd.DataFrame) -> None:
    """Channel mix by quarter and each channel's week by half-year (normal days that are not special)."""
    q = panel.assign(q=panel.date.dt.to_period("Q").astype(str)).groupby(["store", "q"])[
        ["delivery", "pickup", "sales"]].sum()
    mix = pd.DataFrame({"delivery_share": q.delivery / q.sales, "pickup_share": q.pickup / q.sales}).reset_index()
    mix.to_csv(OUT / "channels_mix.csv", index=False, float_format="%.4f")
    n = panel[panel.normal & (panel.special == "")].copy()
    n["half"] = n.date.dt.year.astype(str) + "H" + np.where(n.date.dt.month > 6, "2", "1")
    rows = []
    for c in ["sales", *COLS]:
        g = n[n[c] > 0].groupby(["store", "half", "dow"])[c].mean().unstack()
        g = g.div(g.mean(axis=1), axis=0) * 100
        g.columns = DAYS
        g["weekend_premium"] = (g[["Sat", "Sun"]].mean(axis=1) / g[DAYS[:5]].mean(axis=1) - 1) * 100
        rows.append(g.assign(channel=c).reset_index())
    pd.concat(rows).to_csv(OUT / "channels_weekday.csv", index=False, float_format="%.1f")


def errors(days: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.concat([shape_errors(g, col) for _, g in days.groupby("origin")])


def table(days: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        z = errors(days, m)
        kind = np.where(z.special.isin(DAYS_OFF), "holiday", np.where(z.special != "", "other special",
                                                                     np.where(z.dow >= 5, "weekend", "weekday")))
        r = {"method": m, "all": z.err.sum() / z.sales.sum() * 100}
        for k in ("weekday", "weekend", "holiday", "other special"):
            r[k] = z.err[kind == k].sum() / z.sales[kind == k].sum() * 100
        for y in (2025, 2026):
            sel = z.date.dt.year == y
            r[f"months {y}"] = z.err[sel].sum() / z.sales[sel].sum() * 100
        rows.append(r)
    return pd.DataFrame(rows).set_index("method").round(2)


def bootstrap(days: pd.DataFrame, a: str, b: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in daily error (b minus a, points) with a 95% interval, resampling store x calendar-month blocks
    (all origins of a store-month move together)."""
    za, zb = errors(days, a), errors(days, b)
    ga = za.groupby(["store", "month"])[["err", "sales"]].sum()
    gb = zb.groupby(["store", "month"]).err.sum().reindex(ga.index)
    e_a, e_b, s = ga.err.to_numpy(), gb.to_numpy(), ga.sales.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    diff = (e_b[idx].sum(1) - e_a[idx].sum(1)) / s[idx].sum(1) * 100
    return (e_b.sum() - e_a.sum()) / s.sum() * 100, *np.percentile(diff, [2.5, 97.5])


def backtest(panel: pd.DataFrame) -> pd.DataFrame:
    stores = load_stores(LAST_DAY)
    days = pd.read_pickle(OUT / "backtest_days.pkl")
    days = days.merge(panel[["store", "date", *COLS]], on=["store", "date"], how="left")
    trees = {c: channel_tree_panel(panel, c) for c in COLS}
    t0 = time.time()
    for origin, idx in days.groupby("origin").groups.items():
        test = days.loc[idx]
        hist = panel[panel.date <= origin]
        jobs = []
        for first, second in GROUPINGS.values():
            big = mix_last(panel, origin, 12, second) >= MIN_SHARE
            jobs += [(s, c) for s in STORES for c in (first, second)
                     if (test.store == s).any() and (c == first or big[s])]
        res = Parallel(n_jobs=4)(delayed(prophet_fit)(channel_store(stores[s], hist[hist.store == s], c), origin,
                                                      pd.DatetimeIndex(test[test.store == s].date), "tuned", "ext")
                                 for s, c in jobs)
        for (s, c), w in zip(jobs, res):
            days.loc[test.index[test.store == s], f"p_{c}"] = w
        for c in COLS:
            days.loc[idx, f"lgbm_{c}"] = LGBM(trees[c], origin, half_life=180).weights(test)
        print(f"origin {origin.date()}: {time.time() - t0:.0f}s", flush=True)
    for c in COLS:
        days[f"blend_{c}"] = np.sqrt(days[f"p_{c}"] * days[f"lgbm_{c}"])

    a = days.store.map(FINAL_SHRINK).fillna(0)
    days["final"] = days.blend_p_hol ** (1 - a) * (days.blend_p_hol > 0)
    for g, (first, second) in GROUPINGS.items():
        # the second channel's share of each store-month: recent (last 3 or 12 months before the origin) or actual
        share = {}
        for k, months in (("mix3", 3), ("mix12", 12)):
            share[k] = pd.Series(np.nan, index=days.index)
            for origin, idx in days.groupby("origin").groups.items():
                share[k].loc[idx] = days.loc[idx, "store"].map(mix_last(panel, origin, months, second)).to_numpy()
        actual = days.groupby(["origin", "store", "month"])[[second, "sales"]].transform("sum")
        share["mixactual"] = actual[second] / actual.sales
        big = share["mix12"] >= MIN_SHARE
        days[f"big_{g}"] = big
        for k, sh in share.items():
            w = np.where(big, combine(days, f"blend_{first}", f"blend_{second}", sh.where(big, 0)), days.blend_p_hol)
            days[f"{g}_{k}"] = w
            days[f"{g}_{k}_bh"] = w ** (1 - a) * (w > 0)
    return days


def chart(panel: pd.DataFrame, days: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(HERE.parent / "report"))
    import charts
    from build_report import copy_assets, find_design_system

    charts.setup(copy_assets(find_design_system(None)))
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4))
    (a1, a2), (a3, a4) = axes

    # 1. delivery's share by quarter (full quarters)
    q = panel[panel.date < "2026-10-01"].assign(q=panel.date.dt.to_period("Q"))
    g = q.groupby(["store", "q"])[["delivery", "sales"]].sum()
    share = (g.delivery / g.sales * 100).unstack("store")
    share = share.loc[pd.Period("2023Q1"):]
    x = np.arange(len(share))
    for s in STORES:
        if s in ("Beverly Hills", "Brentwood"):
            continue
        a1.plot(x, share[s], color=charts.G300, lw=1.1)
    for s, color, lw in (("Brentwood", charts.G900, 1.8), ("Beverly Hills", charts.PINK, 2.6)):
        y = share[s]
        a1.plot(x, y, color=color, lw=lw)
        a1.text(x[-1] + 0.25, y.iloc[-1], f"{NAME.get(s, s)} {y.iloc[-1]:.0f}%", va="center", fontsize=8.5, color=color)
    a1.text(x[-1] + 0.25, share.drop(columns=["Beverly Hills", "Brentwood"]).iloc[-1].median(), "other cafés",
            va="center", fontsize=8.5, color=charts.G500)
    first_q = [i for i, p in enumerate(share.index) if p.quarter == 1]
    a1.set_xticks(first_q, [str(share.index[i].year) for i in first_q])
    a1.set_xlim(-0.3, len(x) + 2.6)
    a1.set_ylim(0, 35)
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    a1.set_title("Delivery's share of sales grew every year", loc="left", fontsize=11, fontweight=600)

    # 2. Beverly Hills' week by channel, 2025 against 2026
    n = panel[(panel.store == "Beverly Hills") & panel.normal & (panel.special == "")]
    for col, ls, lab in (("cafe", "-", "In the café"), ("delivery", "--", "Delivery")):
        for yr, color in ((2025, charts.G700), (2026, charts.PINK)):
            d = n[n.date.dt.year == yr].groupby("dow")[col].mean()
            a2.plot(range(7), d / d.mean() * 100, ls=ls, color=color, lw=2, marker="o", ms=4,
                    label=f"{lab}, {yr}{' (to Oct 14)' if yr == 2026 else ''}")
    a2.axhline(100, color=charts.G300, lw=0.8)
    a2.set_xticks(range(7), DAYS)
    a2.set_ylim(70, 135)
    a2.legend(fontsize=8.5, ncol=2, loc="upper left", frameon=False)
    a2.set_title("Beverly Hills: average day by weekday (week = 100)", loc="left", fontsize=11, fontweight=600)

    # 3. weekend lift by channel, last 12 months
    last = panel[panel.normal & (panel.special == "") & (panel.date > pd.Timestamp(LAST_DAY) - pd.Timedelta(days=365))]
    mix = last.groupby("store")[["delivery", "sales"]].sum()
    order = (mix.delivery / mix.sales).sort_values().index
    ys = np.arange(len(order))
    def lift(d, col):
        m = d[d[col] > 0].groupby(d.dow >= 5)[col].mean()
        return (m[True] / m[False] - 1) * 100

    for i, s in enumerate(order):
        d = last[last.store == s]
        has_delivery = mix.loc[s, "delivery"] >= 0.01 * mix.loc[s, "sales"]  # not La Jolla Cove
        prem = {"cafe": lift(d, "cafe"), "delivery": lift(d, "delivery") if has_delivery else np.nan}
        a3.plot([prem["cafe"], prem["delivery"]], [i, i], color=charts.G300, lw=1.2, zorder=1)
        a3.plot(prem["cafe"], i, "o", color=charts.G700, ms=7, mec="white", zorder=2)
        a3.plot(prem["delivery"], i, "o", color=charts.PINK, ms=7, mec="white", zorder=3)
    a3.axvline(0, color=charts.G400, lw=0.8)
    a3.set_yticks(ys, [f"{NAME.get(s, s)} ({mix.delivery[s] / mix.sales[s] * 100:.0f}%)" for s in order], fontsize=8.5)
    a3.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
    a3.legend([plt.Line2D([], [], marker="o", color=charts.G700, ls="none"),
               plt.Line2D([], [], marker="o", color=charts.PINK, ls="none")], ["In the café", "Delivery"],
              fontsize=8.5, loc="upper right", frameon=False)
    a3.set_title("Weekend day against a weekday, last 12 months\n(in brackets: delivery's share)", loc="left",
                 fontsize=11, fontweight=600)

    # 4. backtest: daily error by store, current model against the channel split
    rows = [("All stores", None)] + [(NAME.get(s, s), s) for s in STORES]
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[0] += 0.5
    for y, (lab, s) in zip(ys, rows):
        sub = days if s is None else days[days.store == s]
        cur, new = (errors(sub, m) for m in ("final", "ch_mix3_bh"))
        c, b = cur.err.sum() / cur.sales.sum() * 100, new.err.sum() / new.sales.sum() * 100
        a4.plot([c, b], [y, y], color=charts.G300, lw=1.2, zorder=1)
        a4.plot(c, y, "D", color=charts.G700, ms=6.5, mec="white", zorder=2)
        a4.plot(b, y, "o", color=charts.PINK, ms=7.5, mec="white", zorder=3)
        a4.text(max(c, b) + 0.3, y, f"{c:.1f}% to {b:.1f}%", va="center", fontsize=8, color=charts.G700)
    a4.set_yticks(ys, [r[0] for r in rows], fontsize=8.5)
    a4.get_yticklabels()[0].set_fontweight(600)
    a4.set_xlim(0, 17)
    a4.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    a4.legend([plt.Line2D([], [], marker="D", color=charts.G700, ls="none"),
               plt.Line2D([], [], marker="o", color=charts.PINK, ls="none")],
              ["Current daily model", "Café and delivery modelled apart"], fontsize=8.5, loc="lower left",
              frameon=False)
    a4.set_title("Daily error given the true month (backtest)", loc="left", fontsize=11, fontweight=600)
    for ax in axes.flat:
        ax.grid(axis="y" if ax in (a1, a2) else "x", color=charts.G100)
        ax.tick_params(labelsize=8.5)
    fig.text(0.012, 0.988, "Delivery has a week of its own, but modelling it apart does not split the days better",
             ha="left", va="top", fontsize=15, fontweight=600, color=charts.INK)
    fig.text(0.012, 0.952, "Toast net sales by channel. In the café = dine-in, take-out, kiosk and online pickup; "
             "delivery = delivery apps and own delivery. Normal days that are not holidays.", ha="left", va="top",
             fontsize=9.5, color=charts.G700)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.06, hspace=0.38, wspace=0.42)
    fig.savefig(OUT / "channels.png", dpi=160)


def main() -> None:
    t0 = time.time()
    panel = channel_panel()
    describe(panel)
    days = backtest(panel)
    # each channel's own days, given its true month (how hard each is to split)
    for g, (first, second) in GROUPINGS.items():
        big = days[f"big_{g}"]
        for c in (first, second):
            sub = days.assign(sales=days[c], normal=days.normal & (days[c] > 0))[big]
            z, zt = errors(sub, f"blend_{c}"), errors(sub, "blend_p_hol")
            print(f"{c:9s} own daily error {z.err.sum() / z.sales.sum() * 100:5.2f}%, "
                  f"with the total's shape {zt.err.sum() / zt.sales.sum() * 100:5.2f}%")
    methods = ["blend_p_hol", "final"] + [f"{g}_{k}" for g in GROUPINGS for k in
                                           ("mix3", "mix12", "mixactual", "mix3_bh", "mixactual_bh")]
    out = table(days, methods)
    print(out.to_string())
    out.to_csv(OUT / "channels_backtest.csv")
    rows = []
    for m in methods:
        z = errors(days, m)
        g = z.groupby("store")[["err", "sales"]].sum()
        rows.append((g.err / g.sales * 100).rename(m))
    by = pd.concat(rows, axis=1).round(2)
    last = panel[panel.date > pd.Timestamp(LAST_DAY) - pd.Timedelta(days=365)].groupby("store")[COLS["online"] +
                                                                                                 ["sales"]].sum()
    by["delivery share (last 12m)"] = (last.delivery / last.sales * 100).round(1)
    by["pickup share (last 12m)"] = (last.pickup / last.sales * 100).round(1)
    by.index = [NAME.get(s, s) for s in by.index]
    print(by.to_string())
    by.to_csv(OUT / "channels_by_store.csv")
    for a, b in (("blend_p_hol", "ch_mix3"), ("blend_p_hol", "ch_mixactual"), ("final", "ch_mix3_bh"),
                 ("blend_p_hol", "on_mix3"), ("blend_p_hol", "on_mixactual"), ("final", "on_mix3_bh")):
        d, lo, hi = bootstrap(days, a, b)
        print(f"{b} vs {a}: {d:+.2f} points [{lo:+.2f}, {hi:+.2f}]")
    bh = days.store == "Beverly Hills"
    print("Beverly Hills by year of the month:")
    print(table(days[bh], methods)[["all", "months 2025", "months 2026", "holiday"]].to_string())
    chart(panel, days)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
