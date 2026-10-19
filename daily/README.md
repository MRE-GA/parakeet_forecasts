# Daily sales for payroll (2027)

The monthly and annual figures for 2027 were already sent (the run-rate review and manager-targets deck and its
workbook). Payroll needs the days, so this folder retrains a model whose only job is to divide each month into its
days, and then splits every month in cents so it matches what was sent exactly. Nothing outside this folder is
changed: it reads the repository's data and settings (`src/forecasting.py`), the targets workbook and the deck's
run-rate (`report/build_targets.py`).

## Metric

**Daily error given the true month.** Take each actual month, split it into days with a method's daily weights, and
compare day by day: Σ|actual day − split day| ÷ Σ actual. Because the months are fixed, this is exactly the error the
daily figures add. Days with closures or data problems are left out; holidays the cafés trade on are kept.

## Backtest

Six forecast origins, quarterly from Dec 2024 to Mar 2026. Each method learns only from data up to the origin and
splits each month of the following year (to Sep 2026) into days: 63 months a store, 10 stores (Nautilus left out).

| Daily error, out of sample | All days | Weekdays | Weekends | Days-off holidays | Other special days |
|---|---|---|---|---|---|
| Equal days | 18.2% | 16.6% | 20.4% | 23.3% | 19.2% |
| One weekday pattern (last 52 weeks) | 10.0% | 9.3% | 9.1% | 24.1% | 18.9% |
| Prophet as tuned (the current daily file) | 9.2% | 9.1% | 8.5% | 12.8% | 13.9% |
| Prophet, weekly pattern freed / separate in summer | 9.1% | 8.9% | 8.5–8.6% | 12.7–12.8% | 14.1–14.3% |
| Components: weekday by season, calibrated holidays, time of year | 9.3% | 9.0% | 8.6% | 14.7% | 16.2% |
| Tree model (LightGBM, all stores together) | 8.8% | 8.6% | 8.4% | 12.7% | 12.1% |
| **Blend: Prophet as tuned × tree model (chosen)** | **8.6%** | **8.5%** | **8.0%** | **11.8%** | **11.7%** |
| Blend, days-off holidays set to their usual level | 8.6% | 8.5% | 8.0% | 12.9% | 11.7% |

By store, the blend against Prophet as tuned: La Jolla Cove 15.0% → 13.4%, Newport Beach 9.4% → 7.9%, Coronado 9.5% →
8.6%, Little Italy 11.3% → 10.4%, Juicery 11.1% → 10.7%, Brentwood 9.8% → 9.5%, Carlsbad 6.9% → 6.6%, La Jolla Café
8.1% → 7.8%, Del Mar 6.7% → 6.5%, Beverly Hills 12.1% → 11.9% (`outputs/backtest_summary.csv`,
`outputs/backtest_by_store.png`). The error is flat across horizons (8.4–8.7% from 1 to 12 months ahead), which matters because 2027
is 3–14 months out.

What the tests showed:

* **The weekday pattern depends on the season.** In every store and every year the weekend premium shrinks in the
  school summer break (Jun 15 – Aug 20): Beverly Hills −9% to −12%, Coronado −13% to −16%, La Jolla Cove −14% to −19%,
  Brentwood −3%. The tree model learns this (and the holiday season); Prophet as tuned does not.
* **The weekend premium is drifting down in several stores** (Beverly Hills, Del Mar, Brentwood, La Jolla Café), so a
  pattern learned from the past overstates weekends; Prophet's shrunk weekly pattern happens to offset that. Blending
  the two keeps the weekday bias within ±1.6%.
* **Holidays are the least precise days** (about 12% against 8.6%). Setting days-off holidays to their usual level
  regardless of weekday helped the Fourth of July but hurt Christmas, Easter and Father's Day, so it is not used.
* **Beverly Hills is the hardest store** (about 12%): its weekdays caught up with its weekends in the first half of
  2026 and its weekend premium came back in September. The blend gives an in-between pattern with the seasonal swing
  (weekends strongest in the fall); review it in January with the fourth quarter.
* Most of the remaining error is day-to-day noise (weather, events) that no calendar model captures.

### Prophet with every special day and its own summer weekly pattern (`prophet_variants.py`)

| Daily error, out of sample | All days | Weekdays | Weekends | Days-off holidays | Other special days |
|---|---|---|---|---|---|
| Prophet as tuned | 9.16% | 9.08% | 8.45% | 12.75% | 13.86% |
| + extended special days (Dec 26, Christmas week, Thanksgiving weekend, spring break, Halloween) | 9.08% | 8.92% | 8.45% | 12.80% | 14.24% |
| + weekly pattern of its own in summer | 9.04% | 8.81% | 8.53% | 12.29% | 14.99% |
| + weekly patterns for summer, the holiday season and the rest | 8.97% | 8.80% | 8.56% | 12.55% | 12.70% |
| Blend (Prophet as tuned × tree model), as built | 8.56% | 8.48% | 8.04% | 11.81% | 11.71% |
| Blend with the extended special days | **8.50%** | 8.41% | 8.03% | 11.81% | 11.53% |
| Blend with extended days and seasonal weekly patterns | 8.58% | 8.44% | 8.25% | 11.61% | 11.32% |

Prophet alone gains from both (9.16% to 8.97%), but the tree model already learns the season-by-weekday pattern, so in
the blend the seasonal weekly patterns count it twice and over-weight weekends. The extended special days help the
blend a little (better in 7 of 10 stores; Halloween, the Christmas week and spring break most).

### Seasonality within the month (`intramonth.py`)

In the data, net of each month's level, the store's weekday-by-season pattern and special days, the first three days
of a month sell 0.5% less, days 14–16 0.6% more and the last three days 0.6% less, each within about one standard
error (±0.4%); paydays (the 1st, the 15th) do not stand out and the signs change from store to store
(`outputs/day_of_month_profile.csv`). The backtest agrees:

| Daily error, out of sample | All days | Days 1–3 | Days 14–16 | Last 3 days |
|---|---|---|---|---|
| Prophet (extended special days) | 9.08% | 10.19% | 8.99% | 9.34% |
| + monthly Fourier seasonality, order 3 / 5, moderate prior | 9.09% / 9.15% | 10.45% / 10.65% | 9.03% / 9.03% | 9.23% / 9.36% |
| + monthly Fourier seasonality, order 5, free prior | 9.12% | 10.60% | 9.00% | 9.32% |
| Tree model / without its day-of-month features | 8.77% / 8.75% | 9.85% / 9.72% | 8.86% / 8.79% | 9.40% / 9.36% |
| Blend (Prophet with extended special days × tree model) | 8.50% | 9.66% | 8.57% | 8.80% |
| Blend with monthly seasonality in Prophet | 8.52–8.54% | 9.77–9.84% | 8.55–8.60% | 8.79–8.83% |
| Blend with an explicit day-of-month adjustment | 8.55% | 9.72% | 8.58% | 8.94% |

No seasonality within the month is worth modelling.

## Output

`outputs/Parakeet_2027_Daily_Forecast.xlsx` and `outputs/daily_2027.csv`: every 2027 day for the ten stores, built by
the blend retrained on all data through Oct 14, 2026, in two series:

* **targets**: each month of 'Targets 2027' (`outputs/Parakeet_2027_Manager_Targets.xlsx`), to the cent;
* **run-rate**: the deck's run-rate objective by month, adding up to the workbook's 'Reference' figure to the cent.

Tabs: the two daily series, average day by weekday and season (a staffing template), a `Check` tab that recomputes
every month against what was sent (zero cents of difference) and notes. The targets are stretch goals (about 5% over
the run-rate for the nine cafés), so payroll built on them assumes that growth is reached.

## Run it

```bash
.venv/bin/python daily/backtest.py       # the methods, six origins -> outputs/backtest_store_month.csv
.venv/bin/python daily/experiments.py    # recency weighting and blend weights for the tree model
.venv/bin/python daily/holiday_test.py   # days-off holidays at their usual level (not adopted)
.venv/bin/python daily/summarize.py      # summary table and chart
.venv/bin/python daily/build_2027.py     # the 2027 daily figures, pinned to what was sent
```
