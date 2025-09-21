SELECT  i.i_item_id,
        i.i_item_desc,
        i.i_category,
        i.i_class,
        i.i_current_price
FROM store_sales AS ss
JOIN item_deep   AS i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim    AS d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE i.i_category IN ('Men', 'Sports', 'Jewelry')
  AND d.d_date BETWEEN DATE '1999-02-05'
                   AND (DATE '1999-02-05' + INTERVAL '30 days')
  AND i.i_embedding <-> 'image_embedding' < 1.30
ORDER BY i.i_embedding <-> 'image_embedding';
