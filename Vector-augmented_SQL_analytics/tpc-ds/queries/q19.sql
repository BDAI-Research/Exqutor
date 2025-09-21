-- Replace :embedding with the query vector text literal, e.g. '[0.1, 0.2, ...]'
SELECT i.i_brand_id   AS brand_id,
       i.i_brand      AS brand,
       i.i_manufact_id,
       i.i_manufact
FROM date_dim    AS d
JOIN store_sales AS ss ON d.d_date_sk = ss.ss_sold_date_sk
JOIN item_deep   AS i  ON ss.ss_item_sk = i.i_item_sk
JOIN customer    AS c  ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address AS ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN store       AS s  ON ss.ss_store_sk = s.s_store_sk
WHERE i.i_manager_id = 14
  AND d.d_moy = 11
  AND d.d_year = 2002
  AND SUBSTR(ca.ca_zip, 1, 5) <> SUBSTR(s.s_zip, 1, 5)
  AND i.i_embedding <-> :embedding < 1.20
ORDER BY i.i_embedding <-> :embedding;
