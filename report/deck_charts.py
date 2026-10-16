"""Charts for the 2027 budget-review deck (inline SVG, Parakeet design system).

Colour roles: actual sales in ink, our plan in Parakeet Pink, the investor budget in mid-gray,
context (ranges, tests, estimates) in light grays.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import re

from charts import G100, G300, G400, G500, G700, G900, INK, PINK, _legend, money_m
from charts import to_svg as _to_svg

BUDGET = G500


def to_svg(fig, name: str) -> str:
    """Inline SVG that scales to the width of its slide column (the tight bounding box makes the
    natural size smaller than figsize, so fixed sizes would leave columns half empty)."""
    svg = _to_svg(fig, name)
    head, rest = svg.split(">", 1)
    head = re.sub(r'\s(width|height)="[^"]*"', "", head)
    # if a page caps the chart's height, shrink it from the top-left corner rather than centring it
    return head + ' width="100%" preserveAspectRatio="xMinYMin meet">' + rest


def esc(text: str) -> str:
    """Escape dollar signs so matplotlib does not read "$a ... $b" as math."""
    return text.replace("$", r"\$")


def money_k(x, _=None):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    return f"${x / 1e3:.0f}K" if x else "$0"


def pct0(x, _=None):
    return f"{x:+.0%}" if x else "0%"


# ------------------------------------------------------------------------------------------
def booked_bars(months, booked, to_come, ours, size=(7.6, 3.3)) -> str:
    """2026 by month: sales already booked (ink), still to come at the budget's estimate (light gray)
    and our own forecast for those months (pink marks). ours: None for months already booked."""
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(months))
    ax.bar(x, booked, width=0.62, color=INK, linewidth=0)
    ax.bar(x, to_come, bottom=booked, width=0.62, color=G300, linewidth=0)
    for i, v in enumerate(ours):
        if v is not None:
            ax.plot([i - 0.4, i + 0.4], [v, v], color=PINK, lw=2.6, solid_capstyle="butt", zorder=4)
    top = max(b + t for b, t in zip(booked, to_come))
    for i, (b, t) in enumerate(zip(booked, to_come)):
        ax.annotate(money_m(b + t), (i, b + t), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8, color=G700)
    ax.set_xticks(x)
    ax.set_xticklabels([m.strftime("%b") for m in months])
    ax.set_ylim(0, top * 1.16)
    ax.set_yticks([])
    ax.grid(False)
    _legend(ax, [Patch(color=INK), Patch(color=G300), Line2D([], [], color=PINK, lw=2.6)],
            ["Booked in Toast (through Oct 14)", "Still to come: budget's estimate", "Our forecast"],
            loc="upper left")
    return to_svg(fig, "booked")


def growth_by_year(labels, values, colors, size=(5.6, 3.3), name="growth-years") -> str:
    """Vertical bars of annual growth rates with value labels."""
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.58, color=colors, linewidth=0)
    for i, v in enumerate(values):
        ax.annotate(f"{v:+.1%}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=10, color=G900, fontweight=600)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_yticks([])
    ax.grid(False)
    return to_svg(fig, name)


def _forecast_lines(ax, n_a, a_last, ours, low, high, budget, n_q4):
    """Our forecast from the last actual month: Oct-Dec 2026 dotted, 2027 plan solid, the range of
    our three models as a band, and the budget (its Oct-Dec 2026 estimate and 2027 objective)."""
    xf = np.arange(n_a - 1, n_a + len(ours))
    band = ax.fill_between(xf, np.r_[a_last, low], np.r_[a_last, high], color=PINK, alpha=0.13, linewidth=0)
    lb = None
    if budget is not None:
        (lb,) = ax.plot(xf, np.r_[a_last, budget], color=BUDGET, lw=1.8, ls=(0, (4, 2.5)))
    (l26,) = ax.plot(xf[: n_q4 + 1], np.r_[a_last, ours[:n_q4]], color=PINK, lw=2.2, ls=(0, (1.1, 1.6)))
    (l27,) = ax.plot(xf[n_q4:], ours[n_q4 - 1:], color=PINK, lw=2.2)
    return xf, band, lb, l26, l27


def _month_axis(ax, months):
    for i, m in enumerate(months):
        if m.month == 1 and i > 0:
            ax.axvline(i - 0.5, color=G100, lw=1, zorder=0)
    idx = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(idx)
    ax.set_xticklabels([months[i].strftime("%b %y") for i in idx])
    ax.set_xlim(-0.5, len(months) - 0.4)


def brand_monthly(months_act, actual, months_fc, ours, low, high, budget, hold_lo, hold_hi, n_q4=3,
                  size=(7.5, 3.55)) -> str:
    """Nine cafés by month: actuals; our forecast (Oct-Dec 2026, then the 2027 plan) with the range of
    our three models; the budget; and a 'no growth after September' band for 2027 (two estimates)."""
    fig, ax = plt.subplots(figsize=size)
    n_a = len(months_act)
    (la,) = ax.plot(np.arange(n_a), actual, color=INK, lw=2)
    xf, band, lb, l26, l27 = _forecast_lines(ax, n_a, actual[-1], ours, low, high, budget, n_q4)
    hold = ax.fill_between(xf[n_q4:], np.r_[ours[n_q4 - 1], hold_lo], np.r_[ours[n_q4 - 1], hold_hi], color=G400,
                           alpha=0.45, linewidth=0)
    months = list(months_act) + list(months_fc)
    _month_axis(ax, months)
    top = max(np.max(high), np.max(actual)) * 1.1
    ax.set_ylim(min(np.min(actual), np.min(hold_lo)) * 0.6, top)
    ax.yaxis.set_major_formatter(FuncFormatter(money_m))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    _legend(ax, [la, l26, l27, band, lb, hold],
            ["Actual", "Our forecast, Oct–Dec 2026", "Our 2027 plan", "Range of our 3 models",
             "Budget: Oct–Dec 2026 and 2027", "No growth after September"],
            loc="upper left", ncol=3, bbox_to_anchor=(0, -0.09))
    return to_svg(fig, "brand-monthly")


def cafe_dumbbell(names, budget_g, actual_g, plan_g, size=(7.4, 4.1)) -> str:
    """Per cafe: 2027 budget growth (gray diamond) vs 2026 growth so far (ink) and our plan (pink)."""
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(names))[::-1]
    for yi, b, a, p in zip(y, budget_g, actual_g, plan_g):
        lo, hi = min(b, a, p), max(b, a, p)
        ax.plot([lo, hi], [yi, yi], color=G100, lw=3, zorder=1, solid_capstyle="butt")
        ax.plot(b, yi, "D", ms=7, color=BUDGET, mec="white", mew=1.5, zorder=3)
        ax.plot(p, yi, "o", ms=8, color=PINK, mec="white", mew=1.5, zorder=4)
        ax.plot(a, yi, "o", ms=8, color=INK, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{b:+.0%}", (b, yi), xytext=(-9, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=G700)
        ax.annotate(f"{a:+.0%}", (max(a, p), yi), xytext=(9, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=8.5, color=INK, fontweight=600)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    allv = list(budget_g) + list(actual_g) + list(plan_g)
    ax.set_xlim(min(0, min(allv)) - 0.06, max(allv) + 0.07)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.axvline(0, color=G300, lw=0.75)
    handles = [Line2D([], [], ls="", marker="D", ms=6, color=BUDGET, mec="white"),
               Line2D([], [], ls="", marker="o", ms=7, color=PINK, mec="white"),
               Line2D([], [], ls="", marker="o", ms=7, color=INK, mec="white")]
    _legend(ax, handles, ["Budget 2027", "Our plan 2027", "Actual Jan–Sep 2026"], loc="lower right", ncol=3,
            bbox_to_anchor=(1.0, 1.02))
    return to_svg(fig, "dumbbell-cafes")


def gap_bars(names, gaps, colors, size=(5.3, 3.9)) -> str:
    """Our plan minus the budget, by cafe ($)."""
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(names))[::-1]
    ax.barh(y, gaps, height=0.58, color=colors, linewidth=0)
    for yi, g in zip(y, gaps):
        # labels sit right of the bar end (or of the zero line for small negatives, clear of the names)
        ax.annotate(esc(f"{'+' if g >= 0 else '−'}${abs(g) / 1e3:.0f}K"), (max(g, 0), yi), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5, color=G900)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0, color=G700, lw=0.75)
    lo = min(0, min(gaps))
    ax.set_xlim(lo * 1.6 - 25_000, max(gaps) * 1.32)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    return to_svg(fig, "gap-bars")


# ------------------------------------------------------------------------------------------
def cafe_monthly(months_act, actual, months_fc, ours, low, high, budget, year_labels, n_q4=3,
                 size=(7.7, 3.55), name="cafe") -> str:
    """One café by month: actuals; our forecast (Oct-Dec 2026 dotted, 2027 plan solid) with the range of
    our three models; and the budget's Oct-Dec 2026 estimate and 2027 objective (None if not budgeted)."""
    fig, ax = plt.subplots(figsize=size)
    n_a = len(months_act)
    (la,) = ax.plot(np.arange(n_a), actual, color=INK, lw=2)
    _, band, lb, l26, l27 = _forecast_lines(ax, n_a, actual[-1], ours, low, high, budget, n_q4)
    handles = [la, l26, l27, band] + ([lb] if lb is not None else [])
    labels = ["Actual", "Our forecast, Oct–Dec 2026", "Our 2027 plan", "Range of our 3 models"] + (
        ["Budget"] if lb is not None else [])
    months = list(months_act) + list(months_fc)
    ref = [np.max(high), np.max(actual)] + ([np.max(budget)] if budget is not None else [])
    low_ref = [np.min(actual), np.min(low)] + ([np.min(budget)] if budget is not None else [])
    top, bottom = max(ref) * 1.08, min(low_ref) * 0.55
    for (y0, y1, label) in year_labels:
        ax.text((y0 + y1) / 2, 1.03, esc(label), ha="center", va="bottom", fontsize=8.2, color=G700, linespacing=1.4,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_formatter(FuncFormatter(money_k))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    _month_axis(ax, months)
    _legend(ax, handles, labels, loc="upper left", ncol=len(labels), bbox_to_anchor=(0, -0.09), columnspacing=1.1)
    return to_svg(fig, name)


def q4_months(labels, last_year, budget, ours, lo, hi, size=(7.4, 3.6)) -> str:
    """October-December 2026 by month: last year, the budget's estimate and our forecast (with the
    range of our three models as a whisker)."""
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(labels))
    w = 0.25
    series = [("Same month 2025", last_year, G300), ("Budget estimate", budget, BUDGET), ("Our forecast", ours, PINK)]
    handles = []
    for k, (lab, vals, col) in enumerate(series):
        xs = x + (k - 1) * (w + 0.03)
        handles.append(ax.bar(xs, vals, width=w, color=col, linewidth=0))
        for xi, v in zip(xs, vals):
            ax.annotate(f"${v / 1e6:.2f}M", (xi, v), xytext=(0, 5 if k < 2 else 11),
                        textcoords="offset points", ha="center", fontsize=8.2, color=G900 if k else G700)
    xo = x + (w + 0.03)
    ax.vlines(xo, lo, hi, color=INK, lw=1.2, zorder=4)
    for xi, a, b in zip(xo, lo, hi):
        ax.hlines([a, b], xi - 0.05, xi + 0.05, color=INK, lw=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(np.max(hi), np.max(budget)) * 1.22)
    ax.set_yticks([])
    ax.grid(False)
    handles.append(Line2D([], [], color=INK, lw=1.2))
    _legend(ax, handles, [s[0] for s in series] + ["Range of our 3 models"], loc="upper left", ncol=4)
    return to_svg(fig, "q4-months")


def quarterly_growth(labels, values, partial, budget_g, plan_g, size=(6.9, 3.35), name="q") -> str:
    """Growth vs the same quarter a year earlier, with the budget's and our 2027 growth as lines.

    partial: (label, value) for the October-to-date bar, or None. budget_g may be None.
    """
    fig, ax = plt.subplots(figsize=size)
    vals = list(values) + ([partial[1]] if partial else [])
    labs = list(labels) + ([partial[0]] if partial else [])
    x = np.arange(len(vals))
    colors = [INK] * len(values) + ([G500] if partial else [])
    ax.bar(x, vals, width=0.6, color=colors, linewidth=0)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.0%}".replace("-", "−"), (i, v), xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.8, color=G900, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    lines = [(plan_g, PINK, f"Our plan 2027  {plan_g:+.1%}")]
    if budget_g is not None:
        lines.append((budget_g, BUDGET, f"Budget 2027  {budget_g:+.1%}"))
    for v, c, lab in lines:
        ax.axhline(v, color=c, lw=1.6, ls=(0, (4, 2.5)), zorder=4)
    ax.axhline(0, color=G700, lw=0.75)
    allv = vals + [v for v, _, _ in lines]
    hi, lo = max(allv), min(0, min(allv))
    span = hi - lo
    ax.set_ylim(lo - span * (0.14 if lo < 0 else 0.02), hi + span * 0.2)
    # line labels at the right edge, nudged apart when close
    pos = sorted([[v, c, lab] for v, c, lab in lines], key=lambda r: r[0])
    if len(pos) == 2 and pos[1][0] - pos[0][0] < span * 0.09:
        mid = (pos[0][0] + pos[1][0]) / 2
        pos[0][0], pos[1][0] = mid - span * 0.05, mid + span * 0.05
    for v, c, lab in pos:
        ax.annotate(lab, (len(vals) - 0.5, v), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.2, color=c, fontweight=600, annotation_clip=False)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}".replace("-", "−")))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    return to_svg(fig, name)


# ------------------------------------------------------------------------------------------
# Run-rate review and manager targets deck
# ------------------------------------------------------------------------------------------
def _pct_label(v: float) -> str:
    return "0%" if round(v * 100) == 0 else f"{v:+.0%}".replace("-", "−")


def _growth_strip(ax, n_a: int, growth: dict) -> list:
    """Growth vs the same month a year earlier under the monthly chart: actual months (gray bars), the
    run-rate (pink bars) and the budget (diamonds), with the last 45 days as a dot. The scale follows the
    actual months; anything far beyond it (a base month hit by a closure, a remodel) is cut and labelled."""
    ga, gr = np.asarray(growth["actual"], dtype=float), np.asarray(growth["run"], dtype=float)
    gb = np.asarray(growth["budget"], dtype=float) if growth.get("budget") is not None else np.full(len(gr), np.nan)
    xa, xf = np.arange(n_a), np.arange(n_a, n_a + len(gr))
    hist = ga[~np.isnan(ga)]
    allv = np.r_[hist, gr[~np.isnan(gr)], gb[~np.isnan(gb)], [growth["g45"]]]
    cap = max(0.25, 1.5 * np.percentile(hist, 90)) if len(hist) else max(0.25, np.nanmax(allv))
    top = max(cap, allv.max() * 1.08) if allv.max() <= cap * 1.15 else cap
    bottom = min(0.0, allv.min() * 1.25) - 0.04 * (top - min(0.0, allv.min()))
    ax.bar(xa, np.minimum(ga, top), width=0.72, color=G500, linewidth=0, zorder=2)
    ax.bar(xf, np.minimum(gr, top), width=0.72, color=PINK, alpha=0.5, linewidth=0, zorder=2)
    keep = gb <= top
    ax.plot(xf[keep], gb[keep], ls="", marker="D", ms=3.4, color=G900, zorder=3)
    ax.plot([n_a - 0.5], [min(growth["g45"], top)], "o", ms=5.5, color=INK, mec="white", mew=1.1, zorder=4)
    # label bars cut at the top (one label per run of neighbouring months)
    cut = [(x, v) for x, v in zip(np.r_[xa, xf], np.r_[ga, gr]) if not np.isnan(v) and v > top]
    groups = []
    for x, v in cut:
        if groups and x - groups[-1][-1][0] == 1:
            groups[-1].append((x, v))
        else:
            groups.append([(x, v)])
    for g in groups:
        vs = [v for _, v in g]
        txt = _pct_label(vs[0]) if len(g) == 1 else f"{_pct_label(min(vs))} to {_pct_label(max(vs))}"
        ax.annotate(txt, (g[0][0] - 0.45, top), xytext=(-2, -1), textcoords="offset points", ha="right", va="top",
                    fontsize=7, color=G700)
    ax.set_ylim(bottom, top)
    ax.axhline(0, color=G400, lw=0.8, zorder=1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct_label(v)))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_ylabel("vs last year", fontsize=7.6, color=G700, labelpad=6)
    return [Patch(color=G500), Patch(color=PINK, alpha=0.5), Line2D([], [], ls="", marker="D", ms=4, color=G900),
            Line2D([], [], ls="", marker="o", ms=5.5, color=INK, mec="white")]


def runrate_monthly(months_act, actual, months_fc, runrate, budget, year_labels=(), n_q4=3, money="k",
                    label27="Run-rate 2027 (no growth)", growth=None, size=(7.5, 3.75), name="rr") -> str:
    """Actuals by month, then today's run-rate held flat with normal seasons (Oct-Dec 2026 dotted, 2027
    solid) and the budget (its Oct-Dec 2026 estimate and 2027 objective; None if not budgeted).

    growth (optional): dict of 'actual' (per actual month, NaN where there is no full month a year earlier),
    'run' and 'budget' (per forecast month) and 'g45' (the last 45 days), shown as a strip under the chart."""
    if growth is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = plt.figure(figsize=size)
        gs = fig.add_gridspec(2, 1, height_ratios=[2.75, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax)
    n_a = len(months_act)
    (la,) = ax.plot(np.arange(n_a), actual, color=INK, lw=2)
    xf = np.arange(n_a - 1, n_a + len(runrate))
    a_last = actual[-1]
    handles, labels = [la], ["Actual"]
    (l26,) = ax.plot(xf[: n_q4 + 1], np.r_[a_last, runrate[:n_q4]], color=PINK, lw=2.2, ls=(0, (1.1, 1.6)))
    (l27,) = ax.plot(xf[n_q4:], runrate[n_q4 - 1:], color=PINK, lw=2.2)
    handles += [l26, l27]
    labels += ["Run-rate, Oct–Dec 2026", label27]
    if budget is not None:
        (lb,) = ax.plot(xf, np.r_[a_last, budget], color=BUDGET, lw=1.8, ls=(0, (4, 2.5)), zorder=1)
        handles.append(lb)
        labels.append("Budget")
    months = list(months_act) + list(months_fc)
    ref_hi = [np.max(actual), np.max(runrate)] + ([np.max(budget)] if budget is not None else [])
    ref_lo = [np.min(actual), np.min(runrate)] + ([np.min(budget)] if budget is not None else [])
    for (y0, y1, label) in year_labels:
        ax.text((y0 + y1) / 2, 1.03, esc(label), ha="center", va="bottom", fontsize=8.2, color=G700, linespacing=1.4,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(min(ref_lo) * (0.55 if growth is None else 0.7), max(ref_hi) * 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(money_k if money == "k" else money_m))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5 if growth is None else 4))
    _month_axis(ax, months)
    if growth is None:
        _legend(ax, handles, labels, loc="upper left", ncol=len(labels), bbox_to_anchor=(0, -0.09),
                columnspacing=1.1)
        return to_svg(fig, name)
    ax.tick_params(labelbottom=False)
    h2 = _growth_strip(ax2, n_a, growth)
    _month_axis(ax2, months)
    l2 = ["Growth vs last year", "Growth at run-rate", "Growth in the budget",
          f"Last 45 days {_pct_label(growth['g45'])}"]
    if budget is None:
        h2, l2 = h2[:2] + h2[3:], l2[:2] + l2[3:]
    # two rows, filled row by row: the lines on top, the growth strip's marks below
    cols = max(len(handles), len(h2))
    pad = lambda hs, ls: (hs + [Line2D([], [], lw=0)] * (cols - len(hs)), ls + [""] * (cols - len(ls)))  # noqa: E731
    (h1, l1), (h2, l2) = pad(handles, labels), pad(h2, l2)
    order = [i for pair in zip(range(cols), range(cols, 2 * cols)) for i in pair]
    hs, ls = [(h1 + h2)[i] for i in order], [(l1 + l2)[i] for i in order]
    _legend(ax2, hs, ls, loc="upper left", ncol=cols, bbox_to_anchor=(0, -0.42), columnspacing=1.1)
    return to_svg(fig, name)


def quarter_trend(years, actual, estimate, run, budget, target, size=(7.4, 3.7), name="qtrend") -> str:
    """Same-store growth against the same quarter a year earlier, by quarter: actual quarters (gray), the current
    quarter's estimate (light gray), the next year at today's run-rate (pink), with the budget (diamonds) and the
    targets (rings) for the same quarters.

    years: the years shown, four quarters each; actual: growth for the quarters with actual sales; estimate: the
    current quarter; run, budget, target: the four quarters of the last year."""
    vals = np.r_[actual, [estimate], run]
    x = np.arange(len(vals))
    n_a = len(actual)
    fig, ax = plt.subplots(figsize=size)
    colors = [G500] * n_a + [G300] + [PINK] * len(run)
    ax.bar(x, vals, width=0.62, color=colors, linewidth=0, zorder=2)
    top = max(np.max(vals), np.max(target)) * 1.18
    # values on the bars with actual sales and the estimate; the last year's bars carry markers (the table next to
    # the chart gives their values)
    for xi, v in zip(x[:len(vals) - len(run)], vals):
        ax.annotate(_pct_label(v), (xi, max(v, 0)), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.4, color=G900, zorder=6)
    xf = x[-len(run):]
    ax.plot(xf, budget, ls="", marker="D", ms=5, color=G900, zorder=5)
    ax.plot(xf, target, ls="", marker="o", ms=7.5, mfc="white", mec=INK, mew=1.6, zorder=5)
    for i in range(1, len(years)):
        ax.axvline(4 * i - 0.5, color=G100, lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q}" for _ in years for q in (1, 2, 3, 4)], fontsize=8)
    for i, y in enumerate(years):
        ax.text(4 * i + 1.5, -0.13, str(y), ha="center", va="top", fontsize=8.6, color=INK, fontweight=600,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color=G400, lw=0.8, zorder=1)
    ax.set_ylim(min(0.0, np.min(vals)) - 0.01, top)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct_label(v)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(False, axis="x")
    _legend(ax, [Patch(color=G500), Patch(color=G300), Patch(color=PINK),
                 Line2D([], [], ls="", marker="D", ms=5, color=G900),
                 Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=INK, mew=1.6)],
            ["Actual", "Q4 2026 estimate", "2027 at today's run-rate", "Budget", "2027 target"],
            loc="upper left", ncol=5, bbox_to_anchor=(0, -0.2), columnspacing=1.2)
    return to_svg(fig, name)


def target_trend(months, monthly: dict, trend: dict, year_labels=(), money="k", run_label="Today's run-rate",
                 size=(7.4, 4.45), name="tt") -> str:
    """The 2027 target as the continuation of the café's trend.

    Top: monthly sales, from the actual months to the Oct-Dec 2026 estimate (dotted) and the 2027 target (pink),
    with today's run-rate (light gray) and the budget (dashed) for reference. Bottom: the same without the season
    (each month divided by its usual seasonal pattern, as a monthly equivalent), so the target's slope can be read
    against the café's own trend.

    monthly / trend: dicts of 'actual' (per actual month), 'est26' (Oct-Dec 2026), and 'target', 'run', 'budget'
    (per 2027 month); months: the actual months followed by Oct 2026 - Dec 2027."""
    fmt = FuncFormatter(money_k if money == "k" else money_m)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.55], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    n = len(monthly["actual"])
    x26, x27 = np.arange(n - 1, n + 3), np.arange(n + 2, n + 15)
    for a, s in ((ax, monthly), (ax2, trend)):
        act, est = np.asarray(s["actual"], dtype=float), np.asarray(s["est26"], dtype=float)
        a.plot(np.arange(n), act, color=INK, lw=2)
        a.plot(x26, np.r_[act[-1], est], color=INK, lw=1.8, ls=(0, (1.1, 1.6)))
        a.plot(x27, np.r_[est[-1], s["run"]], color=G400, lw=1.5, zorder=1)
        a.plot(x27, np.r_[est[-1], s["budget"]], color=BUDGET, lw=1.6, ls=(0, (4, 2.5)), zorder=1)
        a.plot(x27, np.r_[est[-1], s["target"]], color=PINK, lw=2.5, zorder=3)
        a.yaxis.set_major_formatter(fmt)
        _month_axis(a, months)
    ys = np.r_[monthly["actual"], monthly["est26"], monthly["target"], monthly["run"], monthly["budget"]]
    ax.set_ylim(np.nanmin(ys) * 0.7, np.nanmax(ys) * 1.08)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelbottom=False)
    for (y0, y1, label) in year_labels:
        ax.text((y0 + y1) / 2, 1.03, esc(label), ha="center", va="bottom", fontsize=8.2, color=G700, linespacing=1.4,
                transform=ax.get_xaxis_transform())
    ys = np.r_[trend["actual"], trend["est26"], trend["target"], trend["run"], trend["budget"]]
    lo, hi = np.nanmin(ys), np.nanmax(ys)
    ax2.set_ylim(lo - 0.08 * (hi - lo), hi + 0.2 * (hi - lo))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.grid(True, axis="y")
    ax2.grid(False, axis="x")
    ax2.text(0.005, 0.97, "Without the season (monthly equivalent)", transform=ax2.transAxes, ha="left", va="top",
             fontsize=7.8, color=G700)
    _legend(ax2, [Line2D([], [], color=INK, lw=2), Line2D([], [], color=INK, lw=1.8, ls=(0, (1.1, 1.6))),
                  Line2D([], [], color=PINK, lw=2.5), Line2D([], [], color=G400, lw=1.5),
                  Line2D([], [], color=BUDGET, lw=1.6, ls=(0, (4, 2.5)))],
            ["Actual", "Estimate, Oct–Dec 2026", "2027 target", run_label, "Budget"],
            loc="upper left", ncol=5, bbox_to_anchor=(0, -0.2), columnspacing=1.1)
    return to_svg(fig, name)


def growth_dots(names, series, label_series=-1, size=(7.4, 4.0), name="growth-dots") -> str:
    """Per row, 2027 growth under several views on one axis.

    series: list of (legend label, values, marker, colour); the values of series[label_series] are labelled.
    """
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(names))[::-1]
    allv = np.array([v for _, vals, _, _ in series for v in vals])
    for i, yi in enumerate(y):
        vals = [s[1][i] for s in series]
        ax.plot([min(vals), max(vals)], [yi, yi], color=G100, lw=3, zorder=1, solid_capstyle="butt")
    handles = []
    for k, (label, vals, marker, color) in enumerate(series):
        h = ax.plot(vals, y, ls="", marker=marker, ms=8 if marker == "o" else 7, color=color, mec="white", mew=1.5,
                    zorder=3 + k)[0]
        handles.append(h)
    lab = series[label_series]
    for yi, v in zip(y, lab[1]):
        row = [s[1][list(y).index(yi)] for s in series]
        ax.annotate(f"{v:+.0%}".replace("-", "−"), (max(row), yi), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=lab[3], fontweight=600)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(min(0, allv.min()) - 0.04, allv.max() + 0.07)
    ax.axvline(0, color=G300, lw=0.75)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}".replace("-", "−")))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _legend(ax, handles, [s[0] for s in series], loc="lower right", ncol=len(series), bbox_to_anchor=(1.0, 1.02))
    return to_svg(fig, name)
