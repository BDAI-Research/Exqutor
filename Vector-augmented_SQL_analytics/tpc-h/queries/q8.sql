SELECT
    o.o_orderkey,
    l.l_linenumber,
    ps.ps_partkey,
    ps.ps_suppkey,
    n1.n_name AS customer_nation,
    n2.n_name AS supplier_nation,
    o.o_orderdate,
    l.l_quantity,
    l.l_extendedprice * (1 - l.l_discount) AS line_revenue
FROM part AS p
JOIN lineitem AS l
ON l.l_partkey = p.p_partkey
JOIN partsupp_youtube AS ps
ON ps.ps_partkey = l.l_partkey
    AND ps.ps_suppkey = l.l_suppkey
JOIN supplier AS s
ON s.s_suppkey = l.l_suppkey
JOIN orders AS o
ON o.o_orderkey = l.l_orderkey
JOIN customer AS c
ON c.c_custkey = o.o_custkey
JOIN nation AS n1
ON n1.n_nationkey = c.c_nationkey
JOIN nation AS n2
ON n2.n_nationkey = s.s_nationkey
JOIN region AS r
ON r.r_regionkey = n1.n_regionkey
WHERE r.r_name = 'AMERICA'
    AND o.o_orderdate >= DATE '1996-01-01'
    AND o.o_orderdate < DATE '1996-02-01'
    AND p.p_type LIKE '%BRASS'
    AND array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 100;
