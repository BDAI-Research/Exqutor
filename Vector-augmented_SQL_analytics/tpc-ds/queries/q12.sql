-- Replace :embedding with the query vector text literal, e.g. '[0.1, 0.2, ...]'
SELECT  i.i_item_id,
        i.i_item_desc,
        i.i_category,
        i.i_class,
        i.i_current_price
FROM web_sales AS ws
JOIN item_deep AS i ON ws.ws_item_sk = i.i_item_sk
JOIN date_dim  AS d ON ws.ws_sold_date_sk = d.d_date_sk
WHERE i.i_category IN ('Women', 'Children', 'Books')
  AND d.d_date BETWEEN DATE '2001-02-28'
                   AND (DATE '2001-02-28' + INTERVAL '30 days')
  AND i.i_embedding <-> :embedding < 1.08
ORDER BY i.i_embedding <-> :embedding;
