SELECT
    s_name,
    s_address,
    n_name
FROM
    supplier,
    nation
WHERE
    s_suppkey IN (
        SELECT
            ps_suppkey
        FROM
            partsupp_deep,
            (
                SELECT
                    l_partkey AS agg_partkey,
                    l_suppkey AS agg_suppkey,
                    0.5 * SUM(l_quantity) AS agg_quantity
                FROM
                    lineitem
                WHERE
                    l_shipdate >= DATE '1993-01-01'
                    AND l_shipdate < DATE '1993-01-01' + INTERVAL '1' YEAR
                GROUP BY
                    l_partkey,
                    l_suppkey
            ) agg_lineitem
        WHERE
            agg_partkey = ps_partkey
            AND agg_suppkey = ps_suppkey
            AND ps_partkey IN (
                SELECT
                    p_partkey
                FROM
                    part
                WHERE
                    p_name LIKE 'azure%'
            )
            AND ps_availqty > agg_quantity
        and ps_embedding <-> 'image_embedding' < 0.86
    order by
        ps_embedding <-> 'image_embedding'
    )
    and s_nationkey = n_nationkey
    AND n_name = 'MOZAMBIQUE'
order by
    s_name
LIMIT 1;