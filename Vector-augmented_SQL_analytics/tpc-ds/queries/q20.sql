SELECT  i.i_item_id,
        i.i_item_desc,
        i.i_category,
        i.i_class,
        i.i_current_price
FROM catalog_sales AS cs
JOIN item_deep     AS i ON cs.cs_item_sk = i.i_item_sk
JOIN date_dim      AS d ON cs.cs_sold_date_sk = d.d_date_sk
WHERE i.i_category IN ('Books', 'Music', 'Sports')
  AND d.d_date BETWEEN DATE '2002-06-18'
                   AND (DATE '2002-06-18' + INTERVAL '30 days')
  AND i.i_embedding <-> 'image_embedding' < 1.08
ORDER BY i.i_embedding <-> 'image_embedding';
