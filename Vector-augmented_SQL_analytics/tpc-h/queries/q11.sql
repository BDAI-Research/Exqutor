SELECT
    ps.ps_partkey,
    ps.ps_suppkey
FROM partsupp_youtube AS ps,
    supplier AS s,
    nation AS n
WHERE ps.ps_suppkey = s.s_suppkey
    AND s.s_nationkey = n.n_nationkey
    AND n.n_name = 'ARGENTINA'
    AND array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 500;
