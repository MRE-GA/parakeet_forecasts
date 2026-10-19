-- Parakeet Cafe daily net sales per store and channel, from the Toast bronze tables in the Grupo Anderson's BI
-- Postgres (MCP "MCP Andersons", tool run_sql). Same net-sales definition and filters as
-- data/sql/extract_parakeet_daily.sql; each check goes to one channel:
--   in store  every order taken in the café (POS, kiosk, in-store take-out and curbside), invoices
--   pickup    online orders (source = 'API') the guest picks up
--   delivery  orders whose dining option behaves as DELIVERY (UberEats, DoorDash, Grubhub, own delivery, catering)
--
-- Output: one row per store x month; daily values packed as "day=in store/pickup/delivery|..." in whole dollars,
-- plus checksums (n days, tot in cents) that daily/channel_data.py validates. The md5 "pad" column only forces the
-- MCP client to spill the result to a file (no meaning). Run once per window: 2022-08-01..2022-12-31, 2023, 2024,
-- 2025, 2026-01-01..2026-10-14. (The MCP validator rejects any semicolon, hence the '|' separator.)

WITH r AS (SELECT restaurant_guid, venue FROM silver.dim_toast_restaurant WHERE brand = 'Parakeet Cafe'
           AND venue IN ('Beverly Hills','Brentwood','Carlsbad','Carmel Valley','Coronado','La Jolla','La Jolla Cove',
                         'La Jolla Juicery','Little Italy','Newport')),
dopt AS (SELECT guid, max(extra) AS behavior FROM bronze.toast_config WHERE config_type ILIKE '%dining%' GROUP BY guid),
oc AS (
  SELECT c.restaurant_guid, c.business_date, c.check_guid, c.amount,
         CASE WHEN d.behavior = 'DELIVERY' THEN 'D' WHEN o.source = 'API' THEN 'P' ELSE 'S' END AS ch
  FROM bronze.toast_checks c
  JOIN bronze.toast_orders o ON o.restaurant_guid = c.restaurant_guid AND o.order_guid = c.order_guid
  LEFT JOIN dopt d ON d.guid = o.dining_option_guid
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
s AS (
  SELECT oc.restaurant_guid, oc.business_date AS d,
         sum(oc.amount - COALESCE(gc.g, 0)) FILTER (WHERE ch = 'S') AS ins,
         sum(oc.amount - COALESCE(gc.g, 0)) FILTER (WHERE ch = 'P') AS pick,
         sum(oc.amount - COALESCE(gc.g, 0)) FILTER (WHERE ch = 'D') AS dlv
  FROM oc LEFT JOIN gc ON gc.restaurant_guid = oc.restaurant_guid AND gc.check_guid = oc.check_guid
  GROUP BY 1, 2
),
agg AS (
  SELECT r.venue AS v, to_char(d, 'YYYY-MM') AS m, count(*) AS n,
         round(sum(COALESCE(ins, 0) + COALESCE(pick, 0) + COALESCE(dlv, 0)), 2) AS tot,
         string_agg(extract(day FROM d)::int || '=' || round(COALESCE(ins, 0))::int || '/' || round(COALESCE(pick, 0))::int
                    || '/' || round(COALESCE(dlv, 0))::int, '|' ORDER BY d) AS x
  FROM s JOIN r USING (restaurant_guid)
  GROUP BY 1, 2
)
SELECT agg.*, repeat(md5(v || m) || md5(m || v), 6) AS pad
FROM agg
ORDER BY v, m
