SELECT i.i_item_desc,
       w.w_warehouse_name,
       d1.d_week_seq
FROM catalog_sales AS cs
JOIN inventory     AS inv ON cs.cs_item_sk = inv.inv_item_sk
JOIN warehouse     AS w   ON w.w_warehouse_sk = inv.inv_warehouse_sk
JOIN item_deep     AS i   ON i.i_item_sk = cs.cs_item_sk
JOIN customer_demographics AS cd ON cs.cs_bill_cdemo_sk = cd.cd_demo_sk
JOIN household_demographics AS hd ON cs.cs_bill_hdemo_sk = hd.hd_demo_sk
JOIN date_dim AS d1 ON cs.cs_sold_date_sk = d1.d_date_sk
JOIN date_dim AS d2 ON inv.inv_date_sk = d2.d_date_sk
JOIN date_dim AS d3 ON cs.cs_ship_date_sk = d3.d_date_sk
LEFT JOIN promotion       AS p  ON cs.cs_promo_sk = p.p_promo_sk
LEFT JOIN catalog_returns AS cr ON cr.cr_item_sk = cs.cs_item_sk
                               AND cr.cr_order_number = cs.cs_order_number
WHERE d1.d_week_seq = d2.d_week_seq
  AND inv.inv_quantity_on_hand < cs.cs_quantity
  AND d3.d_date > d1.d_date + 5
  AND hd.hd_buy_potential = '501-1000'
  AND d1.d_year = 2002
  AND cd.cd_marital_status = 'M'
  AND i.i_embedding <-> 'image_embedding' < 1.08
ORDER BY i.i_embedding <-> 'image_embedding';
