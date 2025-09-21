SELECT
    c_custkey,
    c_name,
    c_acctbal,
    n_name,
    c_address,
    c_phone,
    c_comment
FROM
    customer,
    orders,
    lineitem,
    nation,
    partsupp_deep
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate >= DATE '1993-11-01'
    AND o_orderdate < DATE '1993-11-01' + INTERVAL '3' MONTH
    AND l_returnflag = 'R'
    AND c_nationkey = n_nationkey
    AND ps_partkey = l_partkey
    AND ps_suppkey = l_suppkey
    and ps_embedding <-> 'image_embedding' < 0.86
order by
    ps_embedding <-> 'image_embedding';