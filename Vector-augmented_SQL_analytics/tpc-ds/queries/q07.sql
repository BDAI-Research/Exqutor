SELECT i.i_item_id
FROM store_sales      AS ss
JOIN customer_demographics AS cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
JOIN date_dim         AS d  ON ss.ss_sold_date_sk = d.d_date_sk
JOIN item_deep        AS i  ON ss.ss_item_sk = i.i_item_sk
JOIN promotion        AS p  ON ss.ss_promo_sk = p.p_promo_sk
WHERE cd.cd_gender = 'M'
  AND cd.cd_marital_status = 'M'
  AND cd.cd_education_status = '4 yr Degree'
  AND (p.p_channel_email = 'N' OR p.p_channel_event = 'N')
  AND d.d_year = 2001
  AND i.i_embedding <-> 'image_embedding' < 1.08
ORDER BY i.i_embedding <-> 'image_embedding';
