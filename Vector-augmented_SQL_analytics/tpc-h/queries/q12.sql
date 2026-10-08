SELECT
    o.o_orderkey,
    l.l_linenumber,
    l.l_shipmode,
    o.o_orderpriority
FROM orders AS o
JOIN lineitem AS l
ON o.o_orderkey = l.l_orderkey
JOIN partsupp_youtube AS ps
ON ps.ps_partkey = l.l_partkey
    AND ps.ps_suppkey = l.l_suppkey
WHERE l.l_shipmode IN ('MAIL', 'SHIP')
    AND l.l_commitdate < l.l_receiptdate
    AND l.l_shipdate < l.l_commitdate
    AND l.l_receiptdate >= DATE '1994-01-01'
    AND l.l_receiptdate < DATE '1995-01-01'
    AND array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 500;
