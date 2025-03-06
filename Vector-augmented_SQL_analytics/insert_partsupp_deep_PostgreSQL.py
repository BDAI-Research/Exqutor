
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
from psycopg2.extensions import register_adapter, AsIs
from multiprocessing import Pool, cpu_count

partsupp_csv_path = f'/path/to/partsupp.csv'
vector_bin_path = '/path/to/deep.fbin'

# Connect to the database
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
    """Read *.fbin file that contains float32 vectors."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        print(filename, nvecs, start_idx, dim)
        arr = np.fromfile(f, dtype=np.float32)
    return arr.reshape(nvecs, dim)

def read_partsupp(partsupp_file_path):
    return pd.read_csv(partsupp_file_path, delimiter='|', header=None, engine="pyarrow")


ps = read_partsupp(partsupp_csv_path)
print(ps.shape)

start_index = 0

def process_chunk(args):
    chunk_idx, chunk_size, embedding_file_path, partsupp_df = args
    start_index = chunk_idx * chunk_size
    vectors = read_fbin(embedding_file_path, start_idx=start_index, chunk_size=chunk_size)
    vectors = vectors[:len(partsupp_df)]

    sub_vectors = vectors[:chunk_size]
    processed_vectors = [np.array2string(vector, separator=", ", max_line_width=np.inf) for vector in sub_vectors]
    partsupp_df_chunk = partsupp_df.iloc[start_index:start_index+chunk_size].copy()
    partsupp_df_chunk['embedding'] = processed_vectors
    extras.execute_batch(cur, insert_query, partsupp_df_chunk.to_records(index=False))


chunk_size = 100000
num_chunks = (len(ps) + chunk_size - 1) // chunk_size

with Pool(cpu_count()) as pool:
    pool.map(process_chunk, [(i, chunk_size, vector_bin_path, ps) for i in range(num_chunks)])

# Close the connection
cur.close()
conn.close()