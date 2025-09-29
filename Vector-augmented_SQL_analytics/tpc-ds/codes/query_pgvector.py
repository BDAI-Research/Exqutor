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

conn = psycopg2.connect(dbname='tpcds')
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

cur.execute("SET hnsw.ef_search = 400;")

# %%
def read_fbin(filename, start_idx=0, chunk_size=None):
    """Read *.fbin file that contains float32 vectors."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        print(filename, nvecs, start_idx, dim)
        arr = np.fromfile(f, dtype=np.float32)
    return arr.reshape(nvecs, dim)

deep = read_fbin('/home/wns41559/pg/query.public.10K.fbin')

# %%
import re

def get_time(result):
    # Planning Time과 Execution Time 추출을 위한 정규식
    planning_time_pattern = re.compile(r"Planning Time: (\d+\.\d+) ms")
    execution_time_pattern = re.compile(r"Execution Time: (\d+\.\d+) ms")

    # Planning Time과 Execution Time을 추출
    planning_time = 0.0
    execution_time = 0.0
    planning_match = planning_time_pattern.search(result[-2][0])
    if planning_match:
        planning_time = float(planning_match.group(1))

    execution_match = execution_time_pattern.search(result[-1][0])
    if execution_match:
        execution_time = float(execution_match.group(1))

    # Planning Time과 Execution Time 합산
    total_time = planning_time + execution_time

    # print(f"Planning Time: {planning_time} ms")
    # print(f"Execution Time: {execution_time} ms")
    # print(f"Total Time: {total_time} ms")
    return total_time

# %% [markdown]
# # TPC-H query sampling

# %%
def run(query, num):
    q = deep[0].tolist()
    # cur.execute(query)
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
    print(np.mean(time_result), np.std(time_result))
    return time_result

num = 10
total_result = []
cur.execute("SET hnsw.ef_search = 400;")

# %%
print(deep[0].tolist())

# %%
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze 
    SELECT
        i_embedding <-> '{deep[0].tolist()}'
    FROM
        item_deep
    ORDER BY
        i_embedding <-> '{deep[0].tolist()}'
"""

cur.execute(query)
r = cur.fetchall()
print(len(r))
for row in r:
    print(row)

# %%
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze 
    SELECT
        i_embedding <-> '{deep[0].tolist()}'
    FROM
        item_deep
    where 
        i_embedding <-> '{deep[0].tolist()}' < 1.08
    ORDER BY
        i_embedding <-> '{deep[0].tolist()}'
"""

cur.execute(query)
r = cur.fetchall()
print(len(r))
for row in r:
    print(row)

# %%
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
    SELECT
        i_embedding <-> '{deep[0].tolist()}'
    FROM
        item_deep
    where
        i_current_price > 1
        and i_embedding <-> '{deep[0].tolist()}' < 1.08
    ORDER BY
        i_embedding <-> '{deep[0].tolist()}'
"""

cur.execute(query)
r = cur.fetchall()
print(len(r))
for row in r:
    print(row)

# %%
# query 7
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_item_id
    from store_sales, customer_demographics, date_dim, item_deep, promotion
    where ss_sold_date_sk = d_date_sk and
        ss_item_sk = i_item_sk and
        ss_cdemo_sk = cd_demo_sk and
        ss_promo_sk = p_promo_sk and
        cd_gender = 'M' and 
        cd_marital_status = 'M' and
        cd_education_status = '4 yr Degree' and
        (p_channel_email = 'N' or p_channel_event = 'N') and
        d_year = 2001  
    and i_embedding <-> '{deep[0].tolist()}' < 1.08
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%
# query 12
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_item_id
        ,i_item_desc 
        ,i_category 
        ,i_class 
        ,i_current_price
    from	
        web_sales
        ,item_deep 
        ,date_dim
    where 
        ws_item_sk = i_item_sk 
        and i_category in ('Women', 'Children', 'Books')
        and ws_sold_date_sk = d_date_sk
        and d_date between cast('2001-02-28' as date) 
                    and (cast('2001-02-28' as date) + INTERVAL '30 days')
    and i_embedding <-> '{deep[0].tolist()}' < 1.08
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%
# query 19
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_brand_id brand_id, i_brand brand, i_manufact_id, i_manufact
    from date_dim, store_sales, item_deep,customer,customer_address,store
    where d_date_sk = ss_sold_date_sk
        and ss_item_sk = i_item_sk
        and i_manager_id=14
        and d_moy=11
        and d_year=2002
        and ss_customer_sk = c_customer_sk 
        and c_current_addr_sk = ca_address_sk
        and substr(ca_zip,1,5) <> substr(s_zip,1,5) 
        and ss_store_sk = s_store_sk 
        and i_embedding <-> '{deep[0].tolist()}' < 1.2
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%
# query 20
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_item_id
        ,i_item_desc 
        ,i_category 
        ,i_class 
        ,i_current_price
    from	
        catalog_sales
        ,item_deep 
        ,date_dim
    where cs_item_sk = i_item_sk 
        and i_category in ('Books', 'Music', 'Sports')
        and cs_sold_date_sk = d_date_sk
        and d_date between cast('2002-06-18' as date) 
                        and (cast('2002-06-18' as date) + INTERVAL '30 days')

        and i_embedding <-> '{deep[0].tolist()}' < 1.08
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%
# query 37
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_item_id
        ,i_item_desc
        ,i_current_price
    from item_deep, inventory, date_dim, catalog_sales
    where i_current_price between 16 and 16 + 30
        and inv_item_sk = i_item_sk
        and d_date_sk=inv_date_sk
        and d_date between cast('1999-03-27' as date) and (cast('1999-03-27' as date) + INTERVAL '60 days')
        and i_manufact_id in (147,153,211)
        and inv_quantity_on_hand between 100 and 500
        and cs_item_sk = i_item_sk
        and i_embedding <-> '{deep[0].tolist()}' < 1.2
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
# time_result = run(query, num)
# total_result.append(time_result)

for row in r:
    print(row)

# %%
# query 42
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  dt.d_year
        ,i_category_id
        ,i_category
    from 	date_dim dt
        ,store_sales
        ,item_deep
    where dt.d_date_sk = store_sales.ss_sold_date_sk
        and store_sales.ss_item_sk = i_item_sk
        and i_manager_id = 1  	
        and dt.d_moy=11
        and dt.d_year=1998
        and i_embedding <-> '{deep[0].tolist()}' < 1.2
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%
# query 72
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_item_desc
        ,w_warehouse_name
        ,d1.d_week_seq
    from catalog_sales
    join inventory on (cs_item_sk = inv_item_sk)
    join warehouse on (w_warehouse_sk=inv_warehouse_sk)
    join item_deep on (i_item_sk = cs_item_sk)
    join customer_demographics on (cs_bill_cdemo_sk = cd_demo_sk)
    join household_demographics on (cs_bill_hdemo_sk = hd_demo_sk)
    join date_dim d1 on (cs_sold_date_sk = d1.d_date_sk)
    join date_dim d2 on (inv_date_sk = d2.d_date_sk)
    join date_dim d3 on (cs_ship_date_sk = d3.d_date_sk)
    left outer join promotion on (cs_promo_sk=p_promo_sk)
    left outer join catalog_returns on (cr_item_sk = cs_item_sk and cr_order_number = cs_order_number)
    where d1.d_week_seq = d2.d_week_seq
        and inv_quantity_on_hand < cs_quantity 
        and d3.d_date > d1.d_date + 5
        and hd_buy_potential = '501-1000'
        and d1.d_year = 2002
        and cd_marital_status = 'M'
    and i_embedding <-> '{deep[0].tolist()}' < 1.08
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)


# %%
# query 82
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select  i_item_id
       ,i_item_desc
       ,i_current_price
       ,i_manufact_id
    from item_deep, inventory, date_dim, store_sales
    where i_current_price between 9 and 9+30
        and inv_item_sk = i_item_sk
        and d_date_sk=inv_date_sk
        and d_date between cast('2001-06-07' as date) and (cast('2001-06-07' as date) + INTERVAL '30 days')
        and i_manufact_id in (404,19,241,660)
        and inv_quantity_on_hand between 100 and 500
        and ss_item_sk = i_item_sk
        and i_embedding <-> '{deep[0].tolist()}' < 1.08
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
# time_result = run(query, num)
# total_result.append(time_result)


# %%
# query 98
query = f""" /*+          indexscan(item_deep deep_hnsw)      */
    explain analyze
	select i_item_id
      ,i_item_desc 
      ,i_category 
      ,i_class 
      ,i_current_price
    from	
        store_sales
            ,item_deep 
            ,date_dim
    where 
        ss_item_sk = i_item_sk 
        and i_category in ('Men', 'Sports', 'Jewelry')
        and ss_sold_date_sk = d_date_sk
        and d_date between cast('1999-02-05' as date) 
                    and (cast('1999-02-05' as date) + INTERVAL '30 days')
        and i_embedding <-> '{deep[0].tolist()}' < 1.3
    order by i_embedding <-> '{deep[0].tolist()}'
"""
    
time_result = run(query, num)
total_result.append(time_result)

# %%
# np.save(filename, total_result)
print(total_result)

# %%
for t in total_result:
    for tt in t:
        print(tt, end='\t')
    print()

# %%



