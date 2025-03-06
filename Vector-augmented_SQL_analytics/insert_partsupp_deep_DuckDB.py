import duckdb
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import os

# 파일 경로 설정
partsupp_csv_path = "/path/to/partsupp.csv"
vector_bin_path = "/path/to/deep.fbin"
duckdb_database_path = "/path/to/exqutor.duckdb"

def read_fbin(filename, start_idx=0, chunk_size=None):
    """Read *.fbin file that contains float32 vectors."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        print(filename, nvecs, start_idx, dim)
        arr = np.fromfile(f, dtype=np.float32)
    return arr.reshape(nvecs, dim)

def read_partsupp(csv_path):
    return pd.read_csv(csv_path, delimiter='|', header=None, engine="pyarrow")

def setup_duckdb():
    con = duckdb.connect(duckdb_database_path)
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")
    con.execute("DROP TABLE IF EXISTS partsupp;")

    create_table_query = """
    CREATE TABLE partsupp (
        PS_PARTKEY      INTEGER NOT NULL,
        PS_SUPPKEY      INTEGER NOT NULL,
        PS_AVAILQTY     INTEGER,
        PS_SUPPLYCOST   DECIMAL,
        PS_COMMENT      VARCHAR(199),
        ps_embedding    FLOAT[96]
    );
    """
    con.execute(create_table_query)
    con.close()
    print("DuckDB table created successfully.")

def insert_chunk(args):
    start_idx, chunk_size, file_path, partsupp_df = args
    local_con = duckdb.connect(duckdb_database_path)
    
    vectors = read_fbin(file_path, start_idx=start_idx, chunk_size=chunk_size)
    processed_vectors = [np.array2string(vec, separator=", ", max_line_width=np.inf) for vec in vectors]
    
    partsupp_df_chunk = partsupp_df.iloc[start_idx:start_idx + chunk_size].copy()
    partsupp_df_chunk['ps_embedding'] = processed_vectors 
    
    local_con.register("temp_df", partsupp_df_chunk)
    local_con.execute("INSERT INTO partsupp SELECT * FROM temp_df")
    local_con.unregister("temp_df")
    local_con.close()

def parallel_insert(file_path, partsupp_df, chunk_size=100000, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    total_rows = len(partsupp_df)
    start_indices = list(range(0, total_rows, chunk_size))
    
    with Pool(num_workers) as pool:
        pool.map(insert_chunk, [(idx, chunk_size, file_path, partsupp_df) for idx in start_indices])

# DuckDB 설정 및 데이터 로드
setup_duckdb()
ps_data = read_partsupp(partsupp_csv_path)
parallel_insert(vector_bin_path, ps_data, chunk_size=100000, num_workers=cpu_count())

print("Data insertion is complete.")