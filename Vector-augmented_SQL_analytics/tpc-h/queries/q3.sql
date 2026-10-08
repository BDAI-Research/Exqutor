SELECT
    o_orderkey,
    l_linenumber,
    l_commitdate
FROM customer,
    orders,
    lineitem,
    partsupp_youtube
WHERE c_custkey = o_custkey
    AND o_orderkey = l_orderkey
    AND ps_partkey = l_partkey
    AND ps_suppkey = l_suppkey
    AND c_mktsegment = 'HOUSEHOLD'
    AND o_orderdate < DATE '1995-03-14'
    AND l_shipdate > DATE '1995-03-14'
    AND array_distance(ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 500;
