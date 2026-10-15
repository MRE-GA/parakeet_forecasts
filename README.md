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
.venv/bin/python src/hold_scenario.py        # 'no growth after September' scenario for the budget review
.venv/bin/python src/runrate_scenario.py     # 2027 at today's deseasonalized run-rate (last 45 days)
.venv/bin/python report/build_deck.py --ds /path/to/parakeet-cafe-design     # the budget-review deck
```

The report needs the Parakeet Cafe design-system skill (brand fonts, logo, photography; not
committed here) and Node with Playwright/Chromium for HTML-to-PDF.

To refresh the data, run `data/sql/extract_parakeet_daily.sql` through MCP Andersons `run_sql`
for each window listed in the file, then `python src/build_dataset.py <dump files>`.

## Layout

```
data/raw/parakeet_toast_daily.csv   daily net sales and checks by café (14,239 café-days)
data/cfo_budget_2027*.csv           the investor budget, imported from its workbook (src/import_budget.py)
data/sql/                           extraction query
src/forecasting.py                  data prep + the three models (config-driven) + metrics
src/backtest.py                     round 1 rolling-origin back-test (default settings)
src/tune.py                         round 2 hyperparameter/feature tuning with a 2026 holdout + plan factor
src/feature_ablation.py             one-feature-at-a-time ablation (tuning set and holdout)
src/forecast_2027.py                final fit and 2027 forecast (adopted settings)
src/hold_scenario.py                2027 with each café's trend held at its Sep 30, 2026 level
report/                             charts, HTML templates, PDF renderer (report and budget deck)
outputs/                            CSV results and the PDF
```
