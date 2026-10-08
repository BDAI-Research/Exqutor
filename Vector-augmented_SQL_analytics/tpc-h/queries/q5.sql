SELECT
    o.o_orderkey,
    l.l_linenumber,
    ps.ps_partkey,
    ps.ps_suppkey,
    n.n_name AS trade_nation,
    o.o_orderdate,
    l.l_quantity,
    l.l_extendedprice * (1 - l.l_discount) AS line_revenue
FROM customer AS c
JOIN orders AS o
ON o.o_custkey = c.c_custkey
JOIN lineitem AS l
ON l.l_orderkey = o.o_orderkey
JOIN supplier AS s
ON s.s_suppkey = l.l_suppkey
    AND s.s_nationkey = c.c_nationkey
JOIN nation AS n
ON n.n_nationkey = s.s_nationkey
JOIN region AS r
ON r.r_regionkey = n.n_regionkey
JOIN partsupp_youtube AS ps
ON ps.ps_partkey = l.l_partkey
    AND ps.ps_suppkey = l.l_suppkey
WHERE r.r_name = 'MIDDLE EAST'
    AND o.o_orderdate >= DATE '1993-01-01'
    AND o.o_orderdate < DATE '1994-01-01'
    AND array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 500;
