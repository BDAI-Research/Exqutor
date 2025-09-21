SELECT
    l_orderkey,
    o_orderdate,
    o_shippriority
FROM
    customer,
    orders,
    lineitem,
    partsupp_deep
WHERE
    c_mktsegment = 'HOUSEHOLD'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < DATE '1995-03-14'
    AND l_shipdate > DATE '1995-03-14'
    AND ps_partkey = l_partkey
    AND ps_suppkey = l_suppkey
    AND ps_embedding <-> 'image_embedding' < 0.86
ORDER BY
    ps_embedding <-> 'image_embedding'