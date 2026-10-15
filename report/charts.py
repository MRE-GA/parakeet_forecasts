"""Brand-styled charts (inline SVG) for the Parakeet Cafe forecast report.

Colour roles follow the Parakeet design system (ink + one accent):
actuals in ink, the recommended forecast in Parakeet Pink, context in grays.
Text is kept as SVG text (svg.fonttype = none) so the PDF renders it in Josefin Sans.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

INK = "#000000"
PINK = "#ef4b81"
G900 = "#373a3c"
G700 = "#565a5c"
G500 = "#82888a"
G400 = "#b0b3b4"
G300 = "#cccccc"
G100 = "#eceeef"
FONT = "Josefin Sans"


def setup(font_files: list[str]) -> None:
    for f in font_files:
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": FONT,
        "font.size": 9,
        "text.color": G900,
        "axes.edgecolor": G300,
        "axes.linewidth": 0.75,
        "axes.labelcolor": G700,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": G100,
        "grid.linewidth": 0.75,
        "xtick.color": G700,
        "ytick.color": G700,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def to_svg(fig, name: str) -> str:
    plt.rcParams["svg.hashsalt"] = name  # unique element ids per inline chart
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def money_m(x, _=None):
    return f"${x / 1e6:.1f}M" if x else "$0"


def money_m0(x, _=None):
    return f"${x / 1e6:.0f}M" if x else "$0"


def _month_ticks(ax, months, every=3, fmt="%b %y"):
    idx = [i for i, m in enumerate(months) if (m.month - 1) % every == 0]
    ax.set_xticks(idx)
    ax.set_xticklabels([months[i].strftime(fmt) for i in idx])


def _legend(ax, handles, labels, **kw):
    leg = ax.legend(handles, labels, loc=kw.pop("loc", "upper left"), ncol=kw.pop("ncol", len(labels)),
                    fontsize=8.5, handlelength=1.6, columnspacing=1.4, borderaxespad=0, **kw)
    for t in leg.get_texts():
        t.set_color(G900)
    return leg


# ------------------------------------------------------------------------------------------
def history(months, values, openings, size=(6.4, 3.85)) -> str:
    """Monthly brand net sales with the cafe openings that stepped it up."""
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(months))
    ax.plot(x, values, color=INK, lw=2)
    for m, label, (dx, dy, ha) in openings:
        i = months.index(m)
        ax.plot(i, values[i], "o", ms=6, color=PINK, mec="white", mew=1.5, zorder=3)
        ax.annotate(label, (i, values[i]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=8, color=G700)
    ax.plot(x[-1], values[-1], "o", ms=6, color=INK, mec="white", mew=1.5, zorder=3)
    ax.annotate(f"{months[-1].strftime('%b %Y')}\n{money_m(values[-1])}", (x[-1], values[-1]),
                xytext=(8, -2), textcoords="offset points", fontsize=8, color=G900, va="center")
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(money_m))
    ax.set_yticks(np.arange(0, max(values) * 1.18, 500_000))
    _month_ticks(ax, months, every=6)
    ax.set_xlim(-0.5, len(months) + 3)
    return to_svg(fig, "history")


def annual_columns(labels, values, kinds, size=(3.5, 3.85)) -> str:
    """Annual totals; kinds: 'actual' | 'estimate' | 'forecast'."""
    fig, ax = plt.subplots(figsize=size)
    colors = {"actual": INK, "estimate": G500, "forecast": PINK}
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.5, color=[colors[k] for k in kinds], linewidth=0)
    for i, v in enumerate(values):
        ax.annotate(money_m(v), (i, v), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, color=G900, fontweight=600)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_yticks([])
    ax.grid(False)
    return to_svg(fig, "annual")


def accuracy_panels(models, panels, best_label, size=(9.7, 2.75)) -> str:
    """Small multiples of horizontal bars, one panel per error metric (lower is better)."""
    fig, axes = plt.subplots(1, len(panels), figsize=size, sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (title, values) in zip(axes, panels):
        colors = [PINK if m == best_label else G500 for m in models]
        ax.barh(y, values, height=0.5, color=colors, linewidth=0)
        for yi, v in zip(y, values):
            ax.annotate(f"{v:.1%}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=G900)
        ax.set_title(title, fontsize=9, color=G900, loc="left", pad=8, fontweight=600)
        ax.set_xlim(0, max(values) * 1.35)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models)
    fig.subplots_adjust(wspace=0.12)
    return to_svg(fig, "accuracy")


def rehearsal(months, actual, blend, low, high, title, size=(4.75, 3.4)) -> str:
    """Back-test fold: actual vs blended forecast with the spread of the three models."""
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(months))
    band = ax.fill_between(x, low, high, color=PINK, alpha=0.12, linewidth=0)
    (lb,) = ax.plot(x, blend, color=PINK, lw=2)
    (la,) = ax.plot(x, actual, color=INK, lw=2, marker="o", ms=4, mec="white", mew=1)
    lo = min(np.min(actual), np.min(low))
    hi = max(np.max(actual), np.max(high))
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.yaxis.set_major_formatter(FuncFormatter(money_m))
    ax.set_xticks(x[::2])
    ax.set_xticklabels([m.strftime("%b %y") for m in months][::2])
    ax.set_title(title, fontsize=9, color=G900, loc="left", pad=8, fontweight=600)
    _legend(ax, [la, lb, band], ["Actual", "Blended forecast", "Range of the 3 models"])
    return to_svg(fig, "rehearsal-" + title[:12])


def forecast(months_act, actual, months_fc, blend, low, high, size=(9.7, 3.55)) -> str:
    """Actuals into the 2027 forecast (blend) with the spread across the three models."""
    fig, ax = plt.subplots(figsize=size)
    n_a = len(months_act)
    xa = np.arange(n_a)
    xf = np.arange(n_a - 1, n_a + len(months_fc))
    blend_c = np.concatenate([[actual[-1]], blend])
    low_c = np.concatenate([[actual[-1]], low])
    high_c = np.concatenate([[actual[-1]], high])
    band = ax.fill_between(xf, low_c, high_c, color=PINK, alpha=0.12, linewidth=0)
    (lb,) = ax.plot(xf, blend_c, color=PINK, lw=2)
    (la,) = ax.plot(xa, actual, color=INK, lw=2)
    ax.axvline(n_a - 0.5, color=G300, lw=0.75)
    top = max(np.max(high), np.max(actual)) * 1.12
    ax.text(n_a - 0.2, top * 0.985, "Forecast", fontsize=8.5, color=G700, va="top")
    ax.text(n_a - 0.8, top * 0.985, "Actual", fontsize=8.5, color=G700, va="top", ha="right")
    # selective labels: last actual, the 2027 peak and the last forecast month
    months = list(months_act) + list(months_fc)
    for i, v, dy in [(n_a - 1, actual[-1], -15), (n_a + int(np.argmax(blend)), np.max(blend), 10),
                     (n_a + len(blend) - 1, blend[-1], 10)]:
        ax.plot(i, v, "o", ms=6, color=INK if i < n_a else PINK, mec="white", mew=1.5, zorder=4)
        ax.annotate(f"{months[i].strftime('%b %y')}  {money_m(v)}", (i, v), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="center", fontsize=8, color=G900)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(FuncFormatter(money_m))
    ax.set_yticks(np.arange(0, top, 500_000))
    _month_ticks(ax, months, every=3)
    ax.set_xlim(-0.5, len(months) - 0.3)
    _legend(ax, [la, lb, band], ["Actual net sales", "2027 forecast (blend of 3 models)",
                                 "Range: lowest to highest model"], loc="lower left")
    return to_svg(fig, "forecast")


def model_dots(labels, values, reference, reference_label, best_label, size=(4.5, 2.75),
               name="dots") -> str:
    """Annual totals on a common scale (models or scenarios), with 2026E as the reference line."""
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(labels))[::-1]
    lo = min(min(values), reference)
    hi = max(values)
    ax.axvline(reference, color=G400, lw=0.75)
    ax.annotate(reference_label, (reference, y[0] + 0.55), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=G700, va="center")
    for yi, v, lab in zip(y, values, labels):
        ax.plot([reference, v], [yi, yi], color=G100, lw=2, zorder=1)
        ax.plot(v, yi, "o", ms=9, color=PINK if lab == best_label else INK, mec="white", mew=2, zorder=3)
        ax.annotate(f"{money_m(v)}  ({v / reference - 1:+.0%})", (v, yi), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=G900)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.55)
    ax.set_ylim(-0.6, len(labels) - 0.1)
    ax.xaxis.set_major_formatter(FuncFormatter(money_m0))
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    return to_svg(fig, name)


def dumbbells(labels, panels, size=(9.7, 2.5)) -> str:
    """Before -> after per model, one panel per metric (lower is better).

    panels: list of (title, before_values, after_values).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=size, sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (title, before, after) in zip(np.atleast_1d(axes), panels):
        hi = max(max(before), max(after))
        for yi, b, a in zip(y, before, after):
            ax.plot([b, a], [yi, yi], color=G300, lw=2, zorder=1)
            ax.plot(b, yi, "o", ms=8, color=G400, mec="white", mew=2, zorder=2)
            ax.plot(a, yi, "o", ms=9, color=PINK if a < b else INK, mec="white", mew=2, zorder=3)
            ax.annotate(f"{a:.1%}", (a, yi), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8, color=G900)
        ax.set_title(title, fontsize=9, color=G900, loc="left", pad=8, fontweight=600)
        ax.set_xlim(0, hi * 1.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    # legend: gray = before, pink = after (improved)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=G400, mec="white"),
               Line2D([], [], ls="", marker="o", ms=8, color=PINK, mec="white")]
    _legend(axes[-1], handles, ["Round 1 settings", "Tuned settings"], loc="lower right", ncol=2)
    fig.subplots_adjust(wspace=0.1)
    return to_svg(fig, "dumbbells")


def growth_bars(labels, deltas, pcts, size=(4.6, 4.95)) -> str:
    """Dollar growth 2027 vs 2026E by cafe (single series, sorted)."""
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, deltas, height=0.55, color=INK, linewidth=0)
    for yi, d, p in zip(y, deltas, pcts):
        ax.annotate(f"+${d / 1e6:.2f}M  ({p:+.0%})", (d, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=G900)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(deltas) * 1.55)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    return to_svg(fig, "growth")
