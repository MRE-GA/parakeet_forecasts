# Parakeet Cafe: 2027 sales forecast

Net-sales forecast for 2027 for the 11 Parakeet Cafe locations, built from Toast POS data in the
Grupo Anderson's BI warehouse (MCP Andersons). Prophet is compared against two classic
time-series models (SARIMA and Holt-Winters) in a rolling back-test, and the findings are
written up in a Parakeet-branded PDF:
**[`outputs/Parakeet_2027_Sales_Forecast.pdf`](outputs/Parakeet_2027_Sales_Forecast.pdf)**.

## Results (data through Oct 14, 2026)

| 2027 net sales, 11 existing cafés | Value | vs 2026E ($29.05M) |
|---|---|---|
| **Plan (recommended budget, leans low 3:1)** | **$32.54M** | **+12.0%** |
| Best estimate: blend of the 3 tuned models | $32.84M | +13.0% |
| Prophet (tuned) | $33.07M | +13.8% |
| SARIMA (tuned) | $33.23M | +14.4% |
| Holt-Winters (round-1 settings) | $32.22M | +10.9% |

**Plan.** Over-forecasting sales costs more than under-forecasting, so the plan is the best estimate
times a factor (×0.991) that minimises an asymmetric cost where an over-forecast dollar counts 3×.
The factor is measured from the blend's actual misses (monthly brand totals, 4–15 months ahead), so the
models themselves stay tuned purely for accuracy. Café targets that each café beats three months in
four need a deeper shade (about −6%), because single cafés miss by more than the total.

**Round 1 (default settings).** 5 rehearsal forecasts (origins Sep 2024 to Sep 2025), months 4–15
ahead, 393 café-months: Prophet 9.7% café-month WAPE / 3.6% brand MAPE, SARIMA 11.6% / 3.4%,
Holt-Winters 9.4% / 3.3%, blend 9.3% / 2.1%.

**Round 2 (tuning, `src/tune.py`).** 150 Optuna trials for Prophet, every one of 48 SARIMA and 96
Holt-Winters settings, scored only on 2025 targets; then a holdout the search never saw (forecasts
made Jun, Sep and Dec 2025 for Jan–Sep 2026, 198 café-months):

| Holdout Jan–Sep 2026 | Café-month error | Brand monthly error | Decision |
|---|---|---|---|
| Prophet | 9.5% → 9.7% | 3.8% → 2.4% | adopt tuned (stiffer trend, smoother seasonality) |
| SARIMA | 11.3% → 8.5% | 1.9% → 2.7% | adopt tuned (local trend, no log, extended orders) |
| Holt-Winters | 9.7% → 9.4% | 4.1% → 5.5% | keep round 1 (tuned version overfit 2025) |
| Blend | 8.8% → 8.6% | 2.6% → 2.4% | inverse-error weights (≈ equal) |

Features (switched on/off one at a time from the best settings, judged on 2025 *and* 2026):
multiplicative seasonality (Prophet, Holt-Winters) and weekday-weighted trading days (SARIMA) help;
Prophet holiday-period regressors and SARIMA's log transform hurt; Prophet's summer weekly pattern
and outlier capping, and Holt-Winters' trading-day and log options, make no consistent difference. One model per café beats one model on the cafés' total for 2 of 3 models.
Weather was not tested (the environment's network policy blocks the weather archive).

Live check: Oct 1–14, 2026 (after the training data) came in at $1.08M; the best estimate was 1.4%
under it and the plan 2.2% under. New locations (Manhattan Beach, Studio City, Enclave) have no
sales yet and are **not** in the forecast.

## Budget review deck (`outputs/Parakeet_2027_Budget_Review.pdf`)

A non-technical, 38-slide review of the investor budget ("2027 Sales Objectives by Café –
Investors", October 2026), in three parts. Budget figures come from the budget workbook
(`2027_Sales_budget_Investors.xlsx`, not committed), imported by `src/import_budget.py` into
`data/cfo_budget_2027.csv`, `data/cfo_budget_2027_monthly.csv` (2025–2027 by month) and
`data/cfo_budget_2027_comp_trend.csv`. Its actuals match Toast within 0.1% (exceptions: Coronado's 2025
includes a juicery through March 2025; Beverly Hills' Jan–Sep 2026 sales report is $18K below Toast).

* **2026 finish**: our own October–December forecast (the same three models, $6.92M for the nine
  existing cafés) lands within 0.7% of the budget's $6.88M, so the budget's $27.88M finish matches
  ours ($27.93M: Toast to September plus that forecast) within 0.2%, and within 2% café by café.
  79% of the year is booked. The only real gap is scope: Nautilus ($0.49M) is not in the budget.
* **2027**: the budget's +3.4% is about what the nine cafés would sell if growth stopped in
  September (+1% to +4%, two estimates). Our plan is $31.30M (+12.1%) against $28.82M; five cafés
  carry 79% of the $2.48M gap. The budget is bolder than our track record on the two new cafés
  and the Cove remodel. Suggested total company: $35.38M against $33.70M.
* **Café by café**: two pages per café (the numbers, then the evidence), with a verdict each:
  objective too low (Beverly Hills, Carlsbad, Del Mar, Little Italy, Newport Beach), slightly low
  (La Jolla Café), defensible (Brentwood, Coronado), in line (La Jolla Juicery), budget above our
  view (La Jolla Cove) and not in the budget (Nautilus).

Café-level figures behind the deck: `outputs/budget_review_by_cafe.csv`.

## Run-rate review and manager targets (`outputs/Parakeet_2027_Run_Rate_and_Targets.pdf`)

A 32-slide deck in two parts covering the nine existing cafés and La Jolla Cove (Nautilus left out),
with a workbook for the store managers (`outputs/Parakeet_2027_Manager_Targets.xlsx`).

* **Part 1, today's run-rate**: the last 45 days of Toast sales (Aug 31 – Oct 14, 2026), divided by
  each café's seasonal pattern (from its Prophet fit) and held flat (`src/runrate_scenario.py`). A
  management adjustment sets Coronado, La Jolla Juicery and Newport Beach at +1.2%, +1.5% and +1.9% on
  2026 for 2027 (their last 45 days alone give −1.5%, −1.2% and +0.6%), noted on their pages. The nine
  cafés finish 2026 at $27.79M (−0.3% against the budget's $27.88M) and sell $28.89M in 2027 (+0.2%
  against $28.82M; $28.71M without the adjustment): in total the budget is a no-growth plan, but not
  café by café. Setting each café to its own run-rate raises Beverly Hills, Del Mar, Carlsbad and Little
  Italy (+$338K) and lowers Coronado, Newport Beach, La Jolla Café, La Jolla Juicery and Brentwood
  (−$268K). La Jolla Cove runs at its run-rate to September 2027, then at the budget's remodeled $125K a
  month ($868K, in line with the budget). One page per store with what we found. Under each monthly
  chart, a strip shows growth against the same month a year earlier (same-store for the nine): actual
  months, then what the run-rate and the budget imply, with the last 45 days marked.
* **Part 2, 2027 manager targets** (updated October 19, 2026): each café's 2027 objective in the October 15
  budget (“2027 Sales Objectives by Café”, a one-page PDF imported by `src/import_budget_pdf.py` into
  `data/budget_2027_oct.csv`), spread over the months along our most accurate forecast (the blend of the three
  tuned models, unshaded: café-month error 8.6% and brand-month 2.4% on the 2026 holdout), in cents: the forecast
  moved by one percentage per café, the same every month, from −6.5% (Carlsbad) to +2.7% (Brentwood), −2.9% for the
  nine. La Jolla Café and La Jolla Juicery share their line ($4,111,305) in proportion to the forecast. $30.67M for
  the nine, exactly the budget (+10.3% on the run-rate's 2026 finish; 6.2% over the run-rate, from today's level in
  January to 11% over it in December) and $868K for La Jolla Cove (flat on 2026 to September, then $125K a month),
  $31.53M in all. One page per store with monthly and quarterly targets and what each asks over today's level.
  Each page's chart runs from the 2024 actuals to the 2027 target (today's run-rate for reference;
  the deck does not show the forecast itself), with the same history without the season underneath, so the target can be read against the café's
  own trend (seasonal factors from the run-rate's Prophet fit: `outputs/runrate_45d_seasonal_by_month.csv`). Part 1
  is unchanged.
* **Workbook**: `Targets 2027` (by store and month), `Monthly tracking` (managers enter actuals; the
  variance fills in), `Year to date` (against target, months entered only) and `Reference` (2026
  finish and 2027 under the investor budget, today's run-rate, our forecast and the target). Recalculated with
  LibreOffice on build; zero formula errors. The same script writes the 2027 run-rate (financial budget) in the
  layout of the `Targets 2027` tab, one sheet: `outputs/Parakeet_2027_Run_Rate_Budget.xlsx` ($28.89M for the nine,
  +3.9%; $29.76M with La Jolla Cove), the same months as the daily run-rate to the cent.
* **Daily targets** (`report/build_daily_targets.py`, `outputs/Parakeet_2027_Daily_Targets.xlsx` and
  `outputs/targets_2027_daily.csv`): each month's target, and the run-rate by month, split over its days by the
  best daily model of the daily study (`daily/README.md`): Prophet with the extended special days × a tree model
  trained on all stores, retrained through Oct 14, 2026, with Beverly Hills halfway to equal days (backtest daily
  error 8.4%, against 9.2% for Prophet as tuned, the split used before). Amounts are in cents and every month adds
  up exactly to the workbook (a `Check` tab recomputes it: zero cents of difference); a `Weekday averages` tab gives
  the average day by weekday and season for staffing. The same script writes the rest of 2026 at today's run-rate by
  day (`outputs/Parakeet_2026_Daily_Run_Rate.xlsx`, `outputs/runrate_2026_daily.csv`): Oct 15–31, November and
  December as in Part 1, adding up to its 'Oct 15 – Dec 31 at run-rate' to the cent ($5.88M for the ten stores).
  First check, Thursday to Sunday Oct 15–18: the ten stores sold $337.4K against $338.6K (−0.4%).

## Week types for payroll, next 52 weeks (`outputs/Parakeet_Week_Types_Next_52_Weeks.pdf`)

A 64-page deck and a workbook (`outputs/Parakeet_Week_Templates_Next_52_Weeks.xlsx`) for scheduling payroll over the
next 52 weeks (Mon–Sun, Oct 26, 2026 to Oct 24, 2027): each of the ten run-rate stores' weeks grouped into types of
week, and for each type the average week by day and hour, in sales and checks (`weektypes/`,
`report/build_week_types.py`). It replaces the first version on the 2027 calendar (Jan 4, 2027 – Jan 2, 2028), which
left November and December 2026 out and borrowed Jan 1–2, 2028 from 2027.

* **Types** (`weektypes/model.py`): each store's weeks of today's run-rate by day (`outputs/runrate_2026_daily.csv`
  to Dec 31, then `outputs/targets_2027_daily.csv`) are split into types in runs of at least 4 weeks, the first and
  last included, minimising each week's squared log distance to its type plus a cost per change of type (four weeks
  5% off), so close types do not alternate. A store gets as many types as it can, at most four, while each type's
  average week (as its template has it) sells at least 10% more than the one below; never fewer than two. Four types:
  La Jolla Cove before its Oct 1, 2027 reopening (then Remodeled, the run-rate's $125K a month). Three: Coronado (Low
  $64.7K a week, Normal $79.5K, High Jun 28 – Aug 15, 2027 $104.3K), La Jolla Juicery, Little Italy Café. Two:
  Beverly Hills, Brentwood, Carlsbad, Del Mar Café, La Jolla Café and Newport Beach (Beverly Hills 7.2%, Brentwood
  8.8%, Del Mar 8.6% and La Jolla Café 9.8% apart, under the 10% rule). The beach peaks in summer; Brentwood and
  Beverly Hills sell least in summer.
* **Deck**: summary, method, the calendar, the types at a glance, then for each store its year week by week and one
  page per type: sales and checks by hour, Mon–Sun; under it the holidays in its weeks that move their day most (up to
  ten, at least 5%: the run-rate's day against the template's same weekday); at the side, a chart of its weeks
  against the template (one dot per week, the lowest and highest named). The holiday weeks with a template of their own are in an
  appendix (a list in date order, then one page each); in the workbook, the `Holiday weeks` sheet.
* **Holiday weeks**: weeks with a special day (the daily model's list) and spring break count a quarter in the split
  and nothing in a type's level; 21 of them sell 10% more or less than their type and get a template of their own
  (Thanksgiving, Christmas and New Year's 2026 first; busier at the beach, quieter at Brentwood; none at Beverly Hills,
  Carlsbad, Del Mar or La Jolla Café). The others' holidays are listed on their type's page with what they sell
  against the template. * **Hours** (`weektypes/hourly.py`, `weektypes/backtest_hourly.py`): hourly net sales and checks by the hour each
  check was opened, Oct 2024 – Oct 14, 2026 (`data/sql/extract_parakeet_hourly.sql`, unpacked and checked against
  the daily file by `weektypes/hourly_data.py` into `data/raw/parakeet_toast_hourly.csv.gz`). A day's hours follow the
  store's pattern for that weekday (recent days weighing more, half-life 90 days) times its type's seasonal shift:
  each type's share of every hour against all the days of the same 52-week year, averaged over the two full years,
  so the drift from year to year (each year more of the day sold before 11am: Coronado's summer weeks 23% → 26%,
  Newport's 35% → 38%) is not read as a season, and the recent summer months do not give their hours to the winter
  types. In summer La Jolla Juicery, Newport, La Jolla Café, Del Mar and Carlsbad sell more of the day after 2pm
  (+1.3 to +3.2 points in both years); Beverly Hills, Brentwood and Coronado move the other way one year and not the
  next. Half the shift is used: it repeats in direction but not in size. Backtest (hourly error splitting the actual
  day; the year to Oct 14, 2026 learned on the year before / Apr – Oct 2026 learned before): weekday pattern × half the
  shift 21.30% / 20.66% (used), weekday pattern alone 21.33% / 20.69%, with the full shift 21.36% / 20.70%, a pattern
  of its own per type and weekday 21.57%; all days weighing the same 20.88%. The shift does not beat the weekday
  pattern by much because the drift to earlier hours is larger than the seasons. Holidays on their own pattern from
  earlier years 24.9% against 26.0%. Checks: sales over the average check by hour and weekday of the last 12 months
  (15.6% hourly error, about 3% high as prices rise). Hours that opened later (Little Italy to 6pm, Friday to Sunday,
  since March 2026) take their share from the days since. Observed holidays keep their weekday's hours.
* **Templates** (`weektypes/templates.py`): a type's day is the average of its weeks' run-rate on that weekday
  (holiday weeks with their own template left out); days in whole dollars split into hours that add up to the day.
  Applied to the calendar, the templates give back the run-rate of the 52 weeks ($29.45M for the ten) within
  rounding (at most $41 a store; the workbook's `Check` tab). Every week keeps its own run-rate by day
  (`outputs/weektypes/weeks.csv`): a week sells on average 1.8% (Beverly Hills) to 4.1% (Coronado) more or less than
  its template; each type's page lists its weeks with that gap, and the workbook's `Weeks` sheet has their days.
* **Live check** (`weektypes/live_check.py`, `data/sql/extract_parakeet_hourly_flat.sql`): Oct 15–19, 2026, the five
  days after the data: the ten stores sold −0.8% against the run-rate (Beverly Hills −4.5% to La Jolla Juicery +15.0%),
  the hourly split missed by 19.5% (19.9% with the weekday pattern alone; the backtest's 20.7%) and checks came in
  +0.2% against expected.
* **Workbook**: `Calendar` (every week's type by store, with the ISO week), `Weeks` (every week's own sales by day
  against its template), `Types`, one sheet per store with its types' templates (sales, then checks; totals are
  formulas, recalculated with zero errors), `Holiday weeks` (the appendix), `All templates` (long table) and `Check`. CSVs in `outputs/weektypes/`.

## Approach

* **Data**: daily net sales per café from `bronze.toast_checks/orders/selections`, using the same
  definition as `silver.v_toast_sales_daily` (non-voided checks minus Toast gift-card sales).
  The view itself times out through the MCP, so `data/sql/extract_parakeet_daily.sql` filters to
  Parakeet and pulls one year at a time; `src/build_dataset.py` unpacks the result and validates
  every month against SQL checksums.
* **Bottom-up**: every model is fitted per café and summed, so openings are not read as growth.
* **Prophet**: daily sales, multiplicative weekly + yearly seasonality, US holidays plus
  cafe-relevant days (Mother's Day is +59%, Christmas Eve -17%).
* **SARIMA / Holt-Winters**: monthly sales divided by weekday-weighted trading days (so the
  number of weekends per month is accounted for), forecast, then multiplied back.
* **Cleaning**: opening ramps, Tropical Storm Hilary (Aug 2023), the LA fires at Brentwood
  (Jan 2025) and closure days are excluded from training but kept in the actuals used to score.
* **Tuning**: settings chosen on 2025 only; adopted only if they also win on the 2026 holdout
  (score = mean of café-month and brand-month WAPE). See `outputs/tuned_configs.json` and
  `outputs/tuning/`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/backtest.py             # round 1 back-test, default settings
.venv/bin/python src/tune.py --trials 150    # round 2: tuning + 2026 holdout + plan factor -> outputs/tuned_configs.json
.venv/bin/python src/feature_ablation.py     # one-feature-at-a-time check from the best settings
.venv/bin/python src/forecast_2027.py        # 2027 forecast with the adopted settings
.venv/bin/python report/build_report.py --ds /path/to/parakeet-cafe-design   # the PDF
.venv/bin/python -I src/import_budget.py path/to/2027_Sales_budget_Investors.xlsx   # budget workbook -> data/
.venv/bin/python -I src/import_budget_pdf.py path/to/2027_Parakeet_Sales_Budget.pdf   # Oct 2026 budget -> data/
.venv/bin/python src/hold_scenario.py        # 'no growth after September' scenario for the budget review
.venv/bin/python src/runrate_scenario.py     # 2027 at today's deseasonalized run-rate (last 45 days)
.venv/bin/python report/build_deck.py --ds /path/to/parakeet-cafe-design     # the budget-review deck
.venv/bin/python report/build_targets.py --ds /path/to/parakeet-cafe-design  # run-rate review + manager targets (PDF, xlsx)
.venv/bin/python report/build_daily_targets.py   # targets and run-rate by day: 2027, and the rest of 2026 (xlsx, csv)
.venv/bin/python -I weektypes/hourly_data.py dump_store1.txt ... dump_store10.txt   # hourly extracts -> data/raw/
.venv/bin/python weektypes/model.py              # the week types by store (printed)
.venv/bin/python weektypes/backtest_hourly.py    # hourly patterns and average check -> outputs/weektypes/
.venv/bin/python weektypes/templates.py          # templates -> outputs/weektypes/ (csv)
.venv/bin/python -I weektypes/live_check.py dump.txt   # live check on days after the data (flat hourly extract)
.venv/bin/python report/build_week_types.py --ds /path/to/parakeet-cafe-design   # week-types deck + workbook
```

The report needs the Parakeet Cafe design-system skill (brand fonts, logo, photography; not
committed here) and Node with Playwright/Chromium for HTML-to-PDF.

To refresh the data, run `data/sql/extract_parakeet_daily.sql` through MCP Andersons `run_sql`
for each window listed in the file, then `python src/build_dataset.py <dump files>`.

## Layout

```
data/raw/parakeet_toast_daily.csv   daily net sales and checks by café (14,239 café-days)
data/cfo_budget_2027*.csv           the investor budget, imported from its workbook (src/import_budget.py)
data/budget_2027_oct.csv            the October 2026 budget by café, from its PDF (src/import_budget_pdf.py)
data/sql/                           extraction queries (daily; hourly for the week types)
src/forecasting.py                  data prep + the three models (config-driven) + metrics
src/backtest.py                     round 1 rolling-origin back-test (default settings)
src/tune.py                         round 2 hyperparameter/feature tuning with a 2026 holdout + plan factor
src/feature_ablation.py             one-feature-at-a-time ablation (tuning set and holdout)
src/forecast_2027.py                final fit and 2027 forecast (adopted settings)
src/hold_scenario.py                2027 with each café's trend held at its Sep 30, 2026 level
src/import_budget.py                imports the investor budget workbook into data/
src/import_budget_pdf.py            imports the October 2026 budget PDF into data/
src/runrate_scenario.py             2026 finish and 2027 at today's deseasonalized run-rate (last 45 days)
report/build_targets.py             the run-rate review and manager-targets deck, and the targets workbook
report/build_daily_targets.py       the 2027 targets by day (daily model from daily/), adding up to the months to the cent
daily/                              the daily-split study: backtests of the methods, the chosen daily model
data/raw/parakeet_toast_hourly.csv.gz  net sales and checks by café, day and hour (Oct 2024 – Oct 14, 2026)
weektypes/                          week types for payroll: the types, hourly patterns and their backtest, templates
report/build_week_types.py          the week-types deck and its workbook
report/                             charts, HTML templates, PDF renderer (report and all decks)
outputs/                            CSV results and the PDF
```
