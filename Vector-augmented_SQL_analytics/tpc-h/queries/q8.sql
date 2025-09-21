SELECT
    o_year,
    SUM(CASE
        WHEN nation = 'KENYA' THEN volume
        ELSE 0
    END) / SUM(volume) AS mkt_share
FROM
    (
        SELECT
            extract(YEAR FROM o_orderdate) AS o_year,
            l_extendedprice * (1 - l_discount) AS volume,
            n2.n_name AS nation
        FROM
            part,
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2,
            region,
            partsupp_deep
        WHERE
            p_partkey = l_partkey
            AND s_suppkey = l_suppkey
            AND ps_partkey = l_partkey
            AND ps_suppkey = l_suppkey
            AND l_orderkey = o_orderkey
            AND o_custkey = c_custkey
            AND c_nationkey = n1.n_nationkey
            AND n1.n_regionkey = r_regionkey
            AND r_name = 'MIDDLE EAST'
            AND s_nationkey = n2.n_nationkey
            AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
            AND p_type = 'ECONOMY BRUSHED BRASS'
            and ps_embedding <-> 'image_embedding' < 0.86
        ORDER BY
            ps_embedding <-> 'image_embedding'
    ) as all_nations
group by
    o_year
order by
    o_year
LIMIT 1;