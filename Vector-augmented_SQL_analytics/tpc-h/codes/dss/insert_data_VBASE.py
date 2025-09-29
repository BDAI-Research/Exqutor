import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
from psycopg2.extensions import register_adapter, AsIs
from concurrent.futures import ThreadPoolExecutor

partsupp_csv_path = '../../dataset/third_party/tpch-kit/dbgen/partsupp.csv'
deep_bin_path = '../../dataset/DEEP/base.1B.fbin'
wiki_bin_path = '../../dataset/WIKI/base.10M.fbin'

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
    PS_PARTKEY      INTEGER NOT NULL, -- references P_PARTKEY
    PS_SUPPKEY      INTEGER NOT NULL, -- references S_SUPPKEY
    PS_AVAILQTY     INTEGER,
    PS_SUPPLYCOST   DECIMAL,
    PS_COMMENT      VARCHAR(199),
    ps_image_embedding    float8[96],
    ps_text_embedding     float8[768]
);
"""

cur.execute("DROP TABLE IF EXISTS PARTSUPP;")
cur.execute(q)

def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        n_fetch = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=n_fetch * dim, dtype=np.float64, offset=start_idx * 8 * dim) \
            if arr.dtype == np.float64 else np.fromfile(f, count=n_fetch * dim, dtype=np.float32, offset=start_idx * 4 * dim)
    return arr.reshape(-1, dim)

def read_partsupp(partsupp_file_path):
    df = pd.read_csv(partsupp_file_path, delimiter='|', header=None, engine="pyarrow")
    df = df.iloc[:, :5]
    return df

def process_chunk(chunk_idx, chunk_size, image_embedding_file, text_embedding_file, partsupp_df):
    start_index = chunk_idx * chunk_size
    end_index = min(start_index + chunk_size, len(partsupp_df))
    df_chunk = partsupp_df.iloc[start_index:end_index].copy()

    # Read embeddings
    image_vectors = read_fbin(image_embedding_file, start_idx=start_index, chunk_size=(end_index - start_index))
    text_vectors = read_fbin(text_embedding_file, start_idx=start_index, chunk_size=(end_index - start_index))

    df_chunk['ps_image_embedding'] = list(image_vectors)
    df_chunk['ps_text_embedding'] = list(text_vectors)

    # Prepare rows for COPY
    rows = [
        (
            int(row[0]), int(row[1]), int(row[2]), float(row[3]), str(row[4]),
            row[5], row[6]
        )
        for row in df_chunk.itertuples(index=False, name=None)
    ]

    with conn.cursor() as cur:
        copy_sql = """
            COPY PARTSUPP
            (PS_PARTKEY, PS_SUPPKEY, PS_AVAILQTY, PS_SUPPLYCOST, PS_COMMENT, ps_image_embedding, ps_text_embedding)
            FROM STDIN (FORMAT BINARY)
        """
        with cur.copy(copy_sql) as copy:
            copy.set_types(["int4", "int4", "int4", "float8", "text", "float8[]", "float8[]"])
            for row in rows:
                copy.write_row(row)

ps = read_partsupp(partsupp_csv_path)
print(ps.shape)

chunk_size = 10000
num_chunks = (len(ps) + chunk_size - 1) // chunk_size

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(process_chunk, i, chunk_size, deep_bin_path, wiki_bin_path, ps)
        for i in range(num_chunks)
    ]
    for future in futures:
        future.result()

cur.close()
conn.close()