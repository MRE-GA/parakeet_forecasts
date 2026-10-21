-- Parakeet Cafe net sales and checks per store, day and hour, from the Toast bronze tables in the
-- Grupo Anderson's BI Postgres (MCP "MCP Andersons", tool run_sql).
--
-- Same net-sales definition as extract_parakeet_daily.sql (non-voided/non-deleted checks on
-- non-voided/non-deleted orders, minus Toast gift-card sales); the hour is the hour the check was
-- opened, in California time. Net sales are rounded to whole dollars per store, day and hour.
--
-- Output: one row per month; the values packed as "day@hour=net/checks,hour=net/checks,...|day@..."
-- plus checksums (n cells, tot, tchk) that weektypes/hourly_data.py validates.
-- The MCP client cuts results at 120,000 characters, so run it once per store (:store is the Toast
-- venue: Beverly Hills, Brentwood, Carlsbad, Carmel Valley, Coronado, La Jolla, La Jolla Cove,
-- La Jolla Juicery, Little Italy, Newport) for 2024-10-01..2026-10-14 (about 98,000 characters at most).
-- (The MCP validator rejects any semicolon, hence the '|' separator.)

WITH r AS (SELECT restaurant_guid, venue FROM silver.dim_toast_restaurant
           WHERE brand = 'Parakeet Cafe' AND venue = :store),
oc AS (
  SELECT c.restaurant_guid, c.business_date, c.check_guid, c.amount,
         extract(hour FROM (c.opened_date AT TIME ZONE 'America/Los_Angeles'))::int AS hr
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
cell AS (
  SELECT oc.business_date AS d, oc.hr,
         round(sum(oc.amount) - COALESCE(sum(gc.g), 0))::bigint AS net, count(*) AS chk
  FROM oc LEFT JOIN gc ON gc.restaurant_guid = oc.restaurant_guid AND gc.check_guid = oc.check_guid
  GROUP BY 1, 2
),
day AS (
  SELECT d, count(*) AS n, sum(net) AS net, sum(chk) AS chk,
         extract(day FROM d)::int || '@' || string_agg(hr || '=' || net || '/' || chk, ',' ORDER BY hr) AS x
  FROM cell GROUP BY 1
)
SELECT :store AS v, to_char(d, 'YYYY-MM') AS m, sum(n) AS n, sum(net) AS tot, sum(chk) AS tchk,
       string_agg(x, '|' ORDER BY d) AS x
FROM day
GROUP BY 2
ORDER BY 2
