select
	l_shipmode
from
    orders,
    lineitem,
    partsupp_deep
where
    o_orderkey = l_orderkey
    and l_shipmode in ('RAIL', 'SHIP')
    and l_commitdate < l_receiptdate
    and l_shipdate < l_commitdate
    and l_receiptdate >= date '1994-01-01'
    and l_receiptdate < date '1994-01-01' + interval '1' year
    and ps_partkey = l_partkey
    and ps_suppkey = l_suppkey
    and ps_embedding <-> 'image_embedding' < 0.86
order by
    ps_embedding <-> 'image_embedding';