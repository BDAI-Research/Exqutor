import duckdb
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

partsupp_csv_path = "../../dataset/third_party/tpch-kit/dbgen/partsupp.csv"
part_csv_path = "../../dataset/third_party/tpch-kit/dbgen/part.csv"
deep_bin_path = "../../dataset/DEEP/base.1B.fbin"
wiki_bin_path = "../../dataset/WIKI/base.10M.fbin"
duckdb_database_path = "../../../../DuckDB/duckdb-vss/build/release/exqutor_tpc-h.duckdb"

def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        n_fetch = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=n_fetch * dim, dtype=np.float32, offset=start_idx * 4 * dim)
    return arr.reshape(-1, dim)

def read_csv(csv_path, columns):
    df = pd.read_csv(csv_path, delimiter='|', header=None, engine="pyarrow")
    df = df.iloc[:, :columns]
    return df

def setup_duckdb():
    con = duckdb.connect(duckdb_database_path)
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")

    create_partsupp_table = """
    CREATE TABLE partsupp (
        ps_partkey      INTEGER NOT NULL,
        ps_suppkey      INTEGER NOT NULL,
        ps_availqty     INTEGER,
        ps_supplycost   DOUBLE,
        ps_comment      VARCHAR(199),
        ps_image_embedding FLOAT[96],
        ps_text_embedding FLOAT[768]
    );
    """
    con.execute(create_partsupp_table)

    create_part_table = """
    CREATE TABLE part (
        p_partkey      INTEGER NOT NULL,
        p_name         VARCHAR(55),
        p_mfgr         VARCHAR(25),
        p_brand        VARCHAR(10),
        p_type         VARCHAR(25),
        p_size         INTEGER,
        p_container    VARCHAR(10),
        p_retailprice  DOUBLE,
        p_comment      VARCHAR(23),
        text_embedding FLOAT[768]
    );
    """
    con.execute(create_part_table)
    con.close()
    print("DuckDB tables created successfully.")

def insert_partsupp_chunk(start_idx, chunk_size, image_file, text_file, partsupp_df):
    local_con = duckdb.connect(duckdb_database_path)
    end_idx = min(start_idx + chunk_size, len(partsupp_df))

    image_vectors = read_fbin(image_file, start_idx=start_idx, chunk_size=(end_idx - start_idx))
    text_vectors = read_fbin(text_file, start_idx=start_idx, chunk_size=(end_idx - start_idx))

    partsupp_df_chunk = partsupp_df.iloc[start_idx:end_idx].copy()
    partsupp_df_chunk['ps_image_embedding'] = list(image_vectors)
    partsupp_df_chunk['ps_text_embedding'] = list(text_vectors)

    local_con.register("temp_df", partsupp_df_chunk)
    local_con.execute("INSERT INTO partsupp SELECT * FROM temp_df")
    local_con.unregister("temp_df")
    local_con.close()

def insert_part_table(part_df, wiki_vectors):
    con = duckdb.connect(duckdb_database_path)
    part_df = part_df.copy()
    part_df['text_embedding'] = list(wiki_vectors)
    con.register("temp_part_df", part_df)
    con.execute("INSERT INTO part SELECT * FROM temp_part_df")
    con.unregister("temp_part_df")
    con.close()

def parallel_insert_partsupp(image_file, text_file, partsupp_df, chunk_size=10000, num_workers=8):
    total_rows = len(partsupp_df)
    start_indices = list(range(0, total_rows, chunk_size))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(insert_partsupp_chunk, idx, chunk_size, image_file, text_file, partsupp_df)
            for idx in start_indices
        ]
        for future in futures:
            future.result()

setup_duckdb()

# PARTSUPP
partsupp_df = read_csv(partsupp_csv_path, 5)
parallel_insert_partsupp(deep_bin_path, wiki_bin_path, partsupp_df, chunk_size=10000, num_workers=8)

# PART
part_df = read_csv(part_csv_path, 9)
wiki_vectors = read_fbin(wiki_bin_path, start_idx=0, chunk_size=len(part_df))
insert_part_table(part_df, wiki_vectors)


con = duckdb.connect(duckdb_database_path)
con.execute("SET hnsw_enable_experimental_persistence = true;")
con.execute("""
    CREATE INDEX partsupp_deep_hnsw ON partsupp 
    USING HNSW (ps_image_embedding) 
    WITH (metric = 'l2sq', ef_construction = 200);
""")
con.close()