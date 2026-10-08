SELECT
    o.o_orderkey,
    l.l_linenumber,
    ps.ps_partkey,
    ps.ps_suppkey,
    n.n_name AS supplier_nation,
    o.o_orderdate,
    l.l_quantity,
    l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity AS line_profit
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
JOIN nation AS n
ON n.n_nationkey = s.s_nationkey
WHERE p.p_name LIKE '%almond%'
    AND array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 500;
