# %% [markdown]
# # Basic

# %%
import pandas as pd
from tqdm import tqdm
import numpy as np
np.random.seed(0)

from rich.traceback import install
install()

# %%
import psycopg2
from psycopg2 import extras

conn = psycopg2.connect(dbname='tpch')
cur = conn.cursor()
conn.autocommit = True

cur.execute("LOAD 'pg_hint_plan';")
cur.execute("set pg_hint_plan.message_level to notice;")
cur.execute("SET pg_hint_plan.enable_hint to on;")
cur.execute("SET pg_hint_plan.debug_print to on;")

from psycopg2.extensions import register_adapter, AsIs

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_int8(numpy_int8):
    return AsIs(numpy_int8)

def addapt_numpy_array(numpy_array):
    return AsIs(list(numpy_array))

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int32, addapt_numpy_int32)
register_adapter(np.int8, addapt_numpy_int8)
register_adapter(np.ndarray, addapt_numpy_array)

cur.execute("SET partsupp_deep_hnsw.ef_search = 400;")

# %%
def read_fbin(filename, start_idx=0, chunk_size=None):
    """Read *.fbin file that contains float32 vectors."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        print(filename, nvecs, start_idx, dim)
        arr = np.fromfile(f, dtype=np.float32)
    return arr.reshape(nvecs, dim)

deep = read_fbin('../../dataset/DEEP/query.public.10K.fbin')

# %%
import re

def get_time(result):
    planning_time_pattern = re.compile(r"Planning Time: (\d+\.\d+) ms")
    execution_time_pattern = re.compile(r"Execution Time: (\d+\.\d+) ms")

    planning_time = 0.0
    execution_time = 0.0
    planning_match = planning_time_pattern.search(result[-2][0])
    if planning_match:
        planning_time = float(planning_match.group(1))

    execution_match = execution_time_pattern.search(result[-1][0])
    if execution_match:
        execution_time = float(execution_match.group(1))

    total_time = planning_time + execution_time

    return total_time

# %% [markdown]
# # TPC-H query sampling

# %%
def run(query, num):
    q = deep[0].tolist()
    cur.execute(query)
    time_result = []
    for i in range(num):
        cur.execute(query)
        r = cur.fetchall()
        for row in r:
            print(row[0])
        time = get_time(r)
        time_result.append(time)
        print()
    print(time_result)
    if len(time_result) > 2:
        trimmed = sorted(time_result)[1:-1]
        print(np.mean(trimmed), np.std(trimmed))
    else:
        print(np.mean(time_result), np.std(time_result))
    return time_result

num = 10
total_result = []
cur.execute("SET hnsw.ef_search = 400;")

# %%
query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
    SELECT
        ps_embedding <-> '{deep[0].tolist()}'
    FROM
        partsupp
    where
        ps_embedding <-> '{deep[0].tolist()}' < 0.925
    ORDER BY
        ps_embedding <-> '{deep[0].tolist()}'
"""

cur.execute(query)
r = cur.fetchall()
print(len(r))
for row in r:
    print(row)

# %%
for r in query.split('\n'):
    print(r)

# %% [markdown]
# ## Q3

# %%
query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
    SELECT
        l_orderkey,
        o_orderdate,
        o_shippriority
    FROM
        customer,
        orders,
        lineitem,
        partsupp
    WHERE
        c_mktsegment = 'HOUSEHOLD'
        AND c_custkey = o_custkey
        AND l_orderkey = o_orderkey
        AND o_orderdate < DATE '1995-03-14'
        AND l_shipdate > DATE '1995-03-14'
        AND ps_partkey = l_partkey
        AND ps_suppkey = l_suppkey
        AND ps_embedding <-> '{deep[0].tolist()}' < 0.925
    ORDER BY
        ps_embedding <-> '{deep[0].tolist()}'
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q5

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
    SELECT
        n_name
    FROM
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region,
        partsupp
    WHERE
        c_custkey = o_custkey
        AND l_orderkey = o_orderkey
        AND l_suppkey = s_suppkey
        AND c_nationkey = s_nationkey
        AND s_nationkey = n_nationkey
        AND n_regionkey = r_regionkey
        AND r_name = 'MIDDLE EAST'
        AND o_orderdate >= DATE '1993-01-01'
        AND o_orderdate < DATE '1993-01-01' + INTERVAL '1' YEAR
        AND ps_partkey = l_partkey
        AND ps_suppkey = l_suppkey
        AND ps_embedding <-> '{deep[0].tolist()}' < 0.925
    ORDER BY
        ps_embedding <-> '{deep[0].tolist()}'
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q8

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
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
                partsupp
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
				and ps_embedding <-> '{deep[0].tolist()}' < 0.925
			ORDER BY
				ps_embedding <-> '{deep[0].tolist()}'
		) as all_nations
	group by
		o_year
	order by
		o_year
	LIMIT 1;
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q9

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
    SELECT
        nation,
        o_year,
        SUM(amount) AS sum_profit
    FROM
        (
            SELECT
                n_name AS nation,
                extract(YEAR FROM o_orderdate) AS o_year,
                l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
            FROM
                part,
                supplier,
                lineitem,
                partsupp,
                orders,
                nation
            WHERE
                s_suppkey = l_suppkey
                AND ps_suppkey = l_suppkey
                AND ps_partkey = l_partkey
                AND p_partkey = l_partkey
                AND o_orderkey = l_orderkey
                AND s_nationkey = n_nationkey
                AND p_name LIKE '%almond%'
				and ps_embedding <-> '{deep[0].tolist()}' < 0.925
			ORDER BY
				ps_embedding <-> '{deep[0].tolist()}'
		) as profit
	group by
		nation,
		o_year
	order by
		nation,
		o_year desc
	LIMIT 1;
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q10

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
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
        partsupp
    WHERE
        c_custkey = o_custkey
        AND l_orderkey = o_orderkey
        AND o_orderdate >= DATE '1993-11-01'
        AND o_orderdate < DATE '1993-11-01' + INTERVAL '3' MONTH
        AND l_returnflag = 'R'
        AND c_nationkey = n_nationkey
        AND ps_partkey = l_partkey
        AND ps_suppkey = l_suppkey
        and ps_embedding <-> '{deep[0].tolist()}' < 0.925
    order by
		ps_embedding <-> '{deep[0].tolist()}'
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q11

# %%
query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
    SELECT
    ps_partkey
FROM
    partsupp,
    supplier,
    nation
WHERE
    ps_suppkey = s_suppkey
    AND s_nationkey = n_nationkey
    AND n_name = 'ARGENTINA'
    AND ps_embedding <-> '{deep[0].tolist()}' < 0.925
    ORDER BY
        ps_embedding <-> '{deep[0].tolist()}'
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q12

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
	select
	l_shipmode
	from
		orders,
		lineitem,
        partsupp
	where
		o_orderkey = l_orderkey
		and l_shipmode in ('RAIL', 'SHIP')
		and l_commitdate < l_receiptdate
		and l_shipdate < l_commitdate
		and l_receiptdate >= date '1994-01-01'
		and l_receiptdate < date '1994-01-01' + interval '1' year
        and ps_partkey = l_partkey
        and ps_suppkey = l_suppkey
        and ps_embedding <-> '{deep[0].tolist()}' < 0.925
    order by
		ps_embedding <-> '{deep[0].tolist()}'
"""

time_result = run(query, num)
total_result.append(time_result)

# %% [markdown]
# ## Q18

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
	SELECT
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice,
        l_quantity
    FROM
        customer,
        orders,
        lineitem,
        partsupp
    WHERE
        o_orderkey IN (
            SELECT
                l_orderkey
            FROM
                lineitem
            GROUP BY
                l_orderkey
            HAVING
                SUM(l_quantity) > 260
        )
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
        AND ps_partkey = l_partkey
        AND ps_suppkey = l_suppkey
        and ps_embedding <-> '{deep[0].tolist()}' < 0.925
    order by
		ps_embedding <-> '{deep[0].tolist()}'
"""

# time_result = run(query, num)
# total_result.append(time_result)

# %% [markdown]
# ## Q20

# %%

query = f""" /*+          indexscan(partsupp partsupp_deep_hnsw)      */
    explain analyze
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
                partsupp,
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
                        p_name LIKE 'almond%'
                )
                AND ps_availqty > agg_quantity
			and ps_embedding <-> '{deep[0].tolist()}' < 0.925
		order by
			ps_embedding <-> '{deep[0].tolist()}'
	)
	and s_nationkey = n_nationkey
	AND n_name = 'ALGERIA'
	order by
		s_name
	LIMIT 1;
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%


# %%
# np.save(filename, total_result)
print(total_result)

# %%
for t in total_result:
    for tt in t:
        print(tt, end='\t')
    print()

# %%



