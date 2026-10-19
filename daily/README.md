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
| Blend: Prophet as tuned × tree model | 8.6% | 8.5% | 8.0% | 11.8% | 11.7% |
| Blend, days-off holidays set to their usual level | 8.6% | 8.5% | 8.0% | 12.9% | 11.7% |
| **Blend: Prophet with extended special days × tree model (chosen)** | **8.5%** | **8.4%** | **8.0%** | **11.8%** | **11.5%** |

By store, the chosen blend against Prophet as tuned: La Jolla Cove 15.0% → 13.2%, Newport Beach 9.4% → 7.8%, Coronado
9.5% → 8.4%, Little Italy 11.3% → 10.4%, Juicery 11.1% → 10.6%, Brentwood 9.8% → 9.6%, Carlsbad 6.9% → 6.5%, La Jolla
Café 8.1% → 7.8%, Del Mar 6.7% → 6.5%, Beverly Hills 12.1% → 12.0% (`outputs/backtest_summary.csv`,
`outputs/backtest_by_store.png`). The error is flat across horizons (8.4–8.7% from 1 to 12 months ahead), which matters
because 2027 is 3–14 months out. Block bootstrap over store-months (95% intervals): the blend against Prophet −0.60
points [−0.75, −0.46]; the extended special days in the blend −0.05 [−0.11, −0.01]; both real.

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
  2026 and its weekend premium came back in September. See below: its days are pulled halfway toward equal days.
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

### A weekly pattern of its own for recent months (`prophet_recent.py`)

Conditional seasonality by period instead of by season: Prophet (with the extended special days) learns one weekly
pattern from the last 12 (or 6) months before the origin and another from earlier data, and the forecast uses the
recent one, so a store whose week is changing (Beverly Hills) is followed without losing the rest of its history.

| Daily error, out of sample | Prophet alone | Blended with the tree model |
|---|---|---|
| Weekly pattern for all years (tuned prior) | 9.08% | 8.50% |
| Recent 12 months / earlier, free prior | 8.87% | 8.49% |
| Recent 6 months / earlier, free prior | 8.88% | 8.46% |
| Recent 12 months / earlier, split by summer, free prior | 8.92% | 8.52% |
| Recent 12 months / earlier, tuned prior | 9.78% | 8.62% |

The best change yet for Prophet alone (9.08% to 8.87%), but in the blend the gain is not significant (−0.01 points
[−0.06, +0.05] and −0.04 [−0.09, +0.02]) and Brentwood gets worse (9.6% to 10.3%), so it is not used.

### Beverly Hills

The models did improve Beverly Hills on 2025 months (Prophet 11.3%, the blend 10.7%), but on 2026 months every model
missed: its weekend premium, about +30% in 2024 and +20–25% in 2025, vanished in 2026, so models trained on the past
over-forecast Saturdays by 16–19% and Sundays by 12–13% and under-forecast Tuesdays by 13–14%, while equal days scored
9.6%. Its days are not noisier than other stores': split with each month's own weekday pattern, it would land at 7.7%
(Brentwood 7.7%, Newport 7.4%); the gap to the models (4.3 points) is the largest of any store. So its shape is
pulled halfway toward equal days (in logs), which helps in both years and on its holidays, where the learned effects
were too strong; for the other stores the same pull costs a lot (8.2% to 11.8%):

| Beverly Hills, daily error | 2025 months | 2026 months | All | Days-off holidays |
|---|---|---|---|---|
| Equal days | 12.0% | 9.6% | 10.6% | 15.2% |
| The blend | 10.7% | 12.9% | 12.0% | 22.2% |
| **The blend, halfway to equal days (used)** | **10.1%** | **10.5%** | **10.3%** | **15.7%** |
| Only its weekday pattern halfway to flat | 10.2% | 10.9% | 10.6% | 20.5% |

Its 2027 weekday pattern (week = 100), school term / summer / fall: Mon 96 / 97 / 95, Sat 103 / 102 / 106, Sun 103 /
100 / 104. Review it in January with the fourth quarter.

### By channel: the café and delivery apart (`channels.py`)

Toast tags every order with its channel (`sql/channels_by_day.sql`, unpacked by `channel_data.py` into
`data/toast_channels_daily.csv`: 13,217 store-days from Sep 2022 to Oct 14, 2026, each within $1.49 of the daily file,
the rounding to whole dollars per channel). In the café = dine-in, take-out, kiosk and invoices; pickup = online orders
picked up; delivery = delivery apps and own delivery.

* **Delivery's share grew every year**: at most 9% of sales in 2023; over the last 12 months Brentwood 27.8%, Beverly
  Hills 24.6%, La Jolla Café 21.3%, Carlsbad 20.5%, Little Italy 19.3%, Del Mar 17.7%, Newport 12.2%, Coronado 8.5%,
  Juicery 5.4%, La Jolla Cove 0%. Beverly Hills peaked at 27.5% in 2026 Q1 and was back to 22% in Q3. Online pickup
  adds 5–16% (none at La Jolla Cove).
* **Delivery has a week of its own.** Over the last 12 months a weekend day sold, against a weekday, in the café (with
  pickup) +8% (Beverly Hills) to +95% (Newport), but by delivery −19% (Beverly Hills) to +35% (Newport), within ±10%
  in five of the nine cafés with delivery. In Beverly Hills online pickup is weekday business too (−45%). The summer
  flattening of the week is the café's: Beverly Hills −19 points in the café against −3 by delivery, Coronado −27
  against −8, Little Italy −26 against +3.
* **Beverly Hills' change in 2026 is in the café.** Its café weekend lift went from +36% (2025 H1) to +6% (2026 H1)
  and +10% (2026 H2 to Oct 14); the total went from +26% to −2%. The rise of delivery (about 18% to 27% of sales)
  explains about 4 of those 28 points.

Each channel then gets the chosen daily model, and a month's days are the two channel shapes weighted by the
month's mix: the mix of the last 3 or 12 months before the origin, or, as a ceiling, the month's actual mix.

| Daily error of the total, out of sample | All days | Weekends | Days-off holidays | 2025 months | 2026 months |
|---|---|---|---|---|---|
| Blend (Prophet with extended special days × tree model) | 8.50% | 8.03% | 11.81% | 8.60% | 8.43% |
| Café (with pickup) and delivery apart, mix of the last 3 months | 8.51% | 7.99% | 11.76% | 8.64% | 8.41% |
| Same, mix of the last 12 months | 8.51% | 8.01% | 11.78% | 8.62% | 8.43% |
| Same, each month's actual mix | 8.50% | 7.95% | 11.83% | 8.65% | 8.39% |
| In the café and online (pickup with delivery) apart, mix of the last 3 months | 8.51% | 7.95% | 11.79% | 8.62% | 8.42% |
| Same, each month's actual mix | 8.50% | 7.91% | 11.85% | 8.62% | 8.40% |
| **Blend, Beverly Hills halfway to equal days (used)** | **8.36%** | **7.80%** | **11.36%** | **8.55%** | **8.22%** |
| Café and delivery apart, Beverly Hills halfway to equal days | 8.40% | 7.79% | 11.42% | 8.61% | 8.23% |
| In the café and online apart, Beverly Hills halfway to equal days | 8.40% | 7.76% | 11.42% | 8.59% | 8.24% |

No gain (block bootstrap: +0.01 points [−0.04, +0.05] against the blend; +0.04 [−0.01, +0.08] and +0.03 [−0.02,
+0.08] against the method used). It helps where delivery is large and weekday-heavy (Brentwood 9.6% to 9.4%, Del Mar
6.5% to 6.4%, La Jolla Café 7.8% to 7.7%) and hurts elsewhere (Newport 7.8% to 8.1%, Little Italy 10.4% to 10.5%,
Coronado 8.4% to 8.5%, Carlsbad 6.5% to 6.6%). Each channel alone is split better by its own model (café 9.1% to
9.0%, delivery 23.9% to 19.4%), but delivery's days are noisy (a 19% error against 9% in the café), the two channels'
misses offset each other in the total, which the blend already learns directly, and the mix moves too slowly to
matter within a month (even the actual mix gives 8.50%). For Beverly Hills the split takes the blend from 12.0% to
11.7% (11.4% with the actual mix); halfway to equal days takes it to 10.3%, with or without the split. Not used: the
2027 daily file stays as built (`outputs/channels.png`, `outputs/channels_backtest.csv`, `channels_by_store.csv`,
`channels_mix.csv`, `channels_weekday.csv`).

## Output

`outputs/Parakeet_2027_Daily_Forecast.xlsx` and `outputs/daily_2027.csv`: every 2027 day for the ten stores, built by
the chosen blend (Prophet with the extended special days × the tree model) retrained on all data through Oct 14, 2026,
in two series:

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
.venv/bin/python daily/prophet_variants.py   # Prophet with extended special days and seasonal weekly patterns
.venv/bin/python daily/intramonth.py     # seasonality within the month (not adopted)
.venv/bin/python daily/prophet_recent.py # a weekly pattern of its own for recent months (not adopted)
.venv/bin/python -I daily/channel_data.py ch_2022.txt ... ch_2026.txt  # the channel extracts (MCP result files)
.venv/bin/python daily/channels.py       # the café and delivery modelled apart (not adopted)
.venv/bin/python daily/summarize.py      # summary table and chart
.venv/bin/python daily/build_2027.py     # the 2027 daily figures, pinned to what was sent
```
