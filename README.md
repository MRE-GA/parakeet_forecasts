# Parakeet Cafe: 2027 sales forecast

Net-sales forecast for 2027 for the 11 Parakeet Cafe locations, built from Toast POS data in the
Grupo Anderson's BI warehouse (MCP Andersons). Prophet is compared against two classic
time-series models (SARIMA and Holt-Winters) in a rolling back-test, and the findings are
written up in a Parakeet-branded PDF:
**[`outputs/Parakeet_2027_Sales_Forecast.pdf`](outputs/Parakeet_2027_Sales_Forecast.pdf)**.

## Results (data through Oct 14, 2026)

| | 2027 net sales | vs 2026E ($29.04M) |
|---|---|---|
| Prophet | $33.02M | +13.7% |
| SARIMA | $34.19M | +17.7% |
| Holt-Winters | $32.22M | +11.0% |
| **Blend (average of 3), recommended** | **$33.14M** | **+14.1%** |

Back-test: 5 rehearsal forecasts (origins Sep 2024 to Sep 2025), each scored on months 4-15
ahead (the same gap as today to 2027), 393 café-months:

| Model | Café-month WAPE | Brand monthly MAPE | Avg. miss on period total | Worst miss |
|---|---|---|---|---|
| Prophet | 9.7% | 3.6% | 3.2% | 6.8% |
| SARIMA | 11.6% | 3.4% | 2.5% | 6.0% |
| Holt-Winters | 9.4% | 3.3% | 3.0% | 5.7% |
| **Blend** | **9.3%** | **2.1%** | **1.3%** | **2.5%** |

Holt-Winters is the best single model (narrowly ahead of Prophet); the blend beats all three on
every measure. Live check: Oct 1-14, 2026 (after the training data) came in at $1.08M and the
blend was 1.7% under it. New locations (Manhattan Beach, Studio City, Enclave) have no sales yet
and are **not** in the forecast.

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

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/backtest.py             # outputs/backtest_*.csv
.venv/bin/python src/prophet_sensitivity.py  # outputs/prophet_sensitivity.csv
.venv/bin/python src/forecast_2027.py        # outputs/forecast_*.csv, october_check.csv
.venv/bin/python report/build_report.py --ds /path/to/parakeet-cafe-design   # the PDF
```

The report needs the Parakeet Cafe design-system skill (brand fonts, logo, photography; not
committed here) and Node with Playwright/Chromium for HTML-to-PDF.

To refresh the data, run `data/sql/extract_parakeet_daily.sql` through MCP Andersons `run_sql`
for each window listed in the file, then `python src/build_dataset.py <dump files>`.

## Layout

```
data/raw/parakeet_toast_daily.csv   daily net sales and checks by café (14,239 café-days)
data/sql/                           extraction query
src/forecasting.py                  data prep + the three models + metrics
src/backtest.py                     rolling-origin back-test
src/prophet_sensitivity.py          Prophet configuration check
src/forecast_2027.py                final fit and 2027 forecast
report/                             charts, HTML template, PDF renderer
outputs/                            CSV results and the PDF
```
