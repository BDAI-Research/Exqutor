SELECT
    ps_partkey
FROM
    partsupp_deep,
    supplier,
    nation
WHERE
    ps_suppkey = s_suppkey
    AND s_nationkey = n_nationkey
    AND n_name = 'ARGENTINA'
    AND ps_embedding <-> 'image_embedding' < 0.86
ORDER BY
    ps_embedding <-> 'image_embedding';