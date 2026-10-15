-- Parakeet Cafe daily net sales per store, from the Toast bronze tables in the
-- Grupo Anderson's BI Postgres (MCP "MCP Andersons", tool run_sql).
--
-- Same net-sales definition as silver.v_toast_sales_daily:
--   net_sales = sum(check.amount) of non-voided/non-deleted checks on non-voided/non-deleted
--               orders, minus Toast gift-card sales (selection_type LIKE 'TOAST_CARD%').
-- The view itself aggregates every Toast restaurant and times out through the MCP, so this
-- query filters to brand = 'Parakeet Cafe' and one date window at a time.
--
-- Output: one row per store x month; daily values packed as "day=net/checks|..." plus
-- checksums (n, tot, tchk) that src/build_dataset.py validates. The md5 "pad" column only
-- forces the MCP client to spill the result to a file (no semantic meaning).
-- Run once per window: 2022-08-01..2023-12-31, 2024, 2025, 2026-01-01..2026-10-14.
-- (The MCP validator rejects any semicolon, hence the '|' separator.)

WITH r AS (SELECT restaurant_guid, venue FROM silver.dim_toast_restaurant WHERE brand = 'Parakeet Cafe'),
oc AS (
  SELECT c.restaurant_guid, c.business_date, c.check_guid, c.amount
  FROM bronze.toast_checks c
  JOIN bronze.toast_orders o ON o.restaurant_guid = c.restaurant_guid AND o.order_guid = c.order_guid
  WHERE c.restaurant_guid IN (SELECT restaurant_guid FROM r)
    AND c.business_date BETWEEN DATE :start_date AND DATE :end_date
    AND NOT o.voided AND NOT COALESCE(o.deleted, false) AND NOT c.voided AND NOT COALESCE(c.deleted, false)
),
gc AS (
  SELECT s.restaurant_guid, s.check_guid, sum(s.price) AS g
  FROM bronze.toast_selections s
  WHERE s.restaurant_guid IN (SELECT restaurant_guid FROM r)
    AND s.business_date BETWEEN DATE :start_date AND DATE :end_date
    AND s.selection_type LIKE 'TOAST_CARD%' AND NOT s.voided
  GROUP BY 1, 2
),
sales AS (
  SELECT oc.restaurant_guid, oc.business_date AS d,
         round(sum(oc.amount) - COALESCE(sum(gc.g), 0), 2) AS net, count(*) AS chk
  FROM oc LEFT JOIN gc ON gc.restaurant_guid = oc.restaurant_guid AND gc.check_guid = oc.check_guid
  GROUP BY 1, 2
),
agg AS (
  SELECT r.venue AS v, to_char(d, 'YYYY-MM') AS m, count(*) AS n, round(sum(net), 2) AS tot, sum(chk) AS tchk,
         string_agg(extract(day FROM d)::int || '=' || net || '/' || chk, '|' ORDER BY d) AS x
  FROM sales JOIN r USING (restaurant_guid)
  GROUP BY 1, 2
)
SELECT agg.*, md5(v || m) || md5(m || v) || md5(v) || md5(m) AS pad
FROM agg
ORDER BY v, m
