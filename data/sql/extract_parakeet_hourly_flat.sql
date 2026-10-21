-- Parakeet Cafe net sales and checks per store, day and hour for a few days, one row each: the same
-- definitions as extract_parakeet_hourly.sql, unpacked (a week of the ten stores is about 70,000 characters,
-- under the MCP client's 120,000). Used for the live check of the week templates (weektypes/live_check.py) on
-- days after the models' data: 2026-10-15..2026-10-19.

WITH r AS (SELECT restaurant_guid, venue FROM silver.dim_toast_restaurant
           WHERE brand = 'Parakeet Cafe' AND venue <> 'Nautilus'),
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
)
SELECT r.venue AS store, oc.business_date AS date, oc.hr AS hour,
       round(sum(oc.amount) - COALESCE(sum(gc.g), 0))::bigint AS net_sales, count(*) AS checks
FROM oc LEFT JOIN gc ON gc.restaurant_guid = oc.restaurant_guid AND gc.check_guid = oc.check_guid
JOIN r ON r.restaurant_guid = oc.restaurant_guid
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
