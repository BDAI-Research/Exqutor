SELECT
    o.o_orderkey,
    l.l_linenumber,
    ps.ps_partkey,
    ps.ps_suppkey,
    c.c_custkey,
    c.c_name,
    n.n_name AS customer_nation,
    c.c_address,
    c.c_phone,
    o.o_orderdate,
    l.l_extendedprice * (1 - l.l_discount) AS returned_line_sales
FROM customer AS c
JOIN orders AS o
ON o.o_custkey = c.c_custkey
JOIN lineitem AS l
ON l.l_orderkey = o.o_orderkey
JOIN nation AS n
ON n.n_nationkey = c.c_nationkey
JOIN partsupp_youtube AS ps
ON ps.ps_partkey = l.l_partkey
    AND ps.ps_suppkey = l.l_suppkey
WHERE o.o_orderdate >= DATE '1993-10-01'
    AND o.o_orderdate < DATE '1994-01-01'
    AND l.l_returnflag = 'R'
    AND array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) < getvariable('threshold')::DOUBLE
ORDER BY array_distance(ps.ps_embedding, getvariable('query_vector')::FLOAT[1024]) ASC
LIMIT 500;
