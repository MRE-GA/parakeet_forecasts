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
    return head + ' width="100%">' + rest


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
def booked_bars(months, booked, to_come, size=(7.6, 3.3)) -> str:
    """2026 by month: sales already booked (ink) and still to come (light gray)."""
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(months))
    ax.bar(x, booked, width=0.62, color=INK, linewidth=0)
    ax.bar(x, to_come, bottom=booked, width=0.62, color=G300, linewidth=0)
    top = max(b + t for b, t in zip(booked, to_come))
    for i, (b, t) in enumerate(zip(booked, to_come)):
        ax.annotate(money_m(b + t), (i, b + t), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color=G700)
    ax.set_xticks(x)
    ax.set_xticklabels([m.strftime("%b") for m in months])
    ax.set_ylim(0, top * 1.16)
    ax.set_yticks([])
    ax.grid(False)
    _legend(ax, [Patch(color=INK), Patch(color=G300)], ["Booked in Toast (through Oct 14)", "Still to come (estimate)"],
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


def brand_monthly(months_act, actual, months_est, estimate, months_fc, plan, budget, hold_lo, hold_hi,
                  size=(7.5, 3.55)) -> str:
    """Nine cafés by month: actuals, Q4-2026 estimate, and 2027 as our plan, the budget and a
    'no growth after September' band (two estimates)."""
    fig, ax = plt.subplots(figsize=size)
    n_a, n_e = len(months_act), len(months_est)
    xa = np.arange(n_a)
    xe = np.arange(n_a - 1, n_a + n_e)
    xf = np.arange(n_a + n_e - 1, n_a + n_e + len(months_fc))
    last = estimate[-1]
    band = ax.fill_between(xf, np.r_[last, hold_lo], np.r_[last, hold_hi], color=G400, alpha=0.45, linewidth=0)
    (lbud,) = ax.plot(xf, np.r_[last, budget], color=G700, lw=1.8, ls=(0, (4, 2.5)))
    (lp,) = ax.plot(xf, np.r_[last, plan], color=PINK, lw=2.2)
    (le,) = ax.plot(xe, np.r_[actual[-1], estimate], color=INK, lw=1.6, ls=(0, (1.5, 2)))
    (la,) = ax.plot(xa, actual, color=INK, lw=2)
    months = list(months_act) + list(months_est) + list(months_fc)
    for i, m in enumerate(months):
        if m.month == 1 and i > 0:
            ax.axvline(i - 0.5, color=G100, lw=1, zorder=0)
    top = max(np.max(plan), np.max(actual)) * 1.12
    ax.set_ylim(min(np.min(actual), np.min(hold_lo)) * 0.6, top)
    ax.yaxis.set_major_formatter(FuncFormatter(money_m))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    idx = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(idx)
    ax.set_xticklabels([months[i].strftime("%b %y") for i in idx])
    ax.set_xlim(-0.5, len(months) - 0.4)
    _legend(ax, [la, le, lp, lbud, band], ["Actual", "Oct–Dec 2026 estimate", "Our 2027 plan", "Budget 2027",
                                           "No growth after September"],
            loc="upper left", ncol=5, bbox_to_anchor=(0, -0.09))
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
def cafe_monthly(months_act, actual, months_est, estimate, months_fc, plan, low, high, budget,
                 year_labels, size=(7.7, 3.55), name="cafe") -> str:
    """One cafe by month: actuals, the Q4-2026 estimate, 2027 plan (+ model range) and the budget."""
    fig, ax = plt.subplots(figsize=size)
    n_a, n_e = len(months_act), len(months_est)
    xa = np.arange(n_a)
    xe = np.arange(n_a - 1, n_a + n_e)
    xf = np.arange(n_a + n_e - 1, n_a + n_e + len(months_fc))
    last_est = estimate[-1]
    band = ax.fill_between(xf, np.r_[last_est, low], np.r_[last_est, high], color=PINK, alpha=0.13, linewidth=0)
    handles, labels = [], []
    if budget is not None:
        (lbud,) = ax.plot(xf, np.r_[last_est, budget], color=BUDGET, lw=1.8, ls=(0, (4, 2.5)))
    (lp,) = ax.plot(xf, np.r_[last_est, plan], color=PINK, lw=2.2)
    (le,) = ax.plot(xe, np.r_[actual[-1], estimate], color=INK, lw=1.6, ls=(0, (1.5, 2)))
    (la,) = ax.plot(xa, actual, color=INK, lw=2)
    handles = [la, le, lp, band] + ([lbud] if budget is not None else [])
    labels = ["Actual", "Oct–Dec 2026 estimate", "Our 2027 plan", "Range of our 3 models"] + (
        ["Budget 2027"] if budget is not None else [])
    months = list(months_act) + list(months_est) + list(months_fc)
    ref = [np.max(high), np.max(actual)] + ([np.max(budget)] if budget is not None else [])
    top = max(ref) * 1.08
    low_ref = [np.min(actual), np.min(low)] + ([np.min(budget)] if budget is not None else [])
    bottom = min(low_ref) * 0.55
    # year separators with the annual totals
    for i, m in enumerate(months):
        if m.month == 1 and i > 0:
            ax.axvline(i - 0.5, color=G100, lw=1, zorder=0)
    for (y0, y1, label) in year_labels:
        ax.text((y0 + y1) / 2, 1.03, esc(label), ha="center", va="bottom", fontsize=8.2, color=G700, linespacing=1.4,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_formatter(FuncFormatter(money_k))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    idx = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(idx)
    ax.set_xticklabels([months[i].strftime("%b %y") for i in idx])
    ax.set_xlim(-0.5, len(months) - 0.4)
    _legend(ax, handles, labels, loc="upper left", ncol=len(labels), bbox_to_anchor=(0, -0.09))
    return to_svg(fig, name)


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
