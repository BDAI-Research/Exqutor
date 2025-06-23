
import pandas as pd
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
from psycopg2.extensions import register_adapter, AsIs
from concurrent.futures import ThreadPoolExecutor

partsupp_csv_path = f'/path/to/partsupp.csv'
vector_bin_path = '/path/to/deep.fbin'

conn = psycopg2.connect()
cur = conn.cursor()
conn.autocommit = True

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

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

# Create the table
q = """
CREATE TABLE PARTSUPP (
	PS_PARTKEY		INTEGER NOT NULL, -- references P_PARTKEY
	PS_SUPPKEY		INTEGER NOT NULL, -- references S_SUPPKEY
	PS_AVAILQTY		INTEGER,
	PS_SUPPLYCOST	DECIMAL,
	PS_COMMENT		VARCHAR(199),
    ps_embedding    vector(96)
);
"""

cur.execute(q)

insert_query = f"INSERT INTO PARTSUPP VALUES (%s, %s, %s, %s, %s, %s)"

def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_partsupp(partsupp_file_path):
    return pd.read_csv(partsupp_file_path, delimiter='|', header=None, engine="pyarrow")

def process_chunk(chunk_idx, chunk_size, embedding_file_path, partsupp_df):
    start_index = chunk_idx * chunk_size
    
    vectors = read_fbin(embedding_file_path, start_idx=start_index, chunk_size=chunk_size)
    sub_vectors = vectors[:chunk_size]
    processed_vectors = [np.array2string(vector, separator=", ", max_line_width=np.inf) for vector in sub_vectors]
    
    partsupp_df_chunk = partsupp_df.iloc[start_index:start_index+chunk_size].copy()
    partsupp_df_chunk['embedding'] = processed_vectors
    
    extras.execute_batch(cur, insert_query, partsupp_df_chunk.to_records(index=False), page_size=1000)

ps = read_partsupp(partsupp_csv_path)
print(ps.shape)

start_index = 0
chunk_size = 10000
num_chunks = (len(ps) + chunk_size - 1) // chunk_size

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_chunk, i, chunk_size, vector_bin_path, ps) for i in range(num_chunks)]
    for future in futures:
        future.result()

cur.close()
conn.close()