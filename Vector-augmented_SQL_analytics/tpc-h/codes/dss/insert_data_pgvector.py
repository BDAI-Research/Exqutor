import pandas as pd
import numpy as np
import psycopg
import pgvector.psycopg
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

# File paths
partsupp_csv_path = '../../dataset/third_party/tpch-kit/dbgen/partsupp.csv'
part_csv_path = '../../dataset/third_party/tpch-kit/dbgen/part.csv'
deep_bin_path = '../../dataset/DEEP/base.1B.fbin'
wiki_bin_path = '../../dataset/WIKI/base.10M.fbin'

# DB connection
conn = psycopg.connect(dbname='tpch')
pgvector.psycopg.register_vector(conn)
conn.autocommit = True

def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        n_fetch = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=n_fetch * dim, dtype=np.float32, offset=start_idx * 4 * dim)
    return arr.reshape(-1, dim)

def read_csv(path, columns):
    df = pd.read_csv(path, delimiter='|', header=None, engine="pyarrow")
    df = df.iloc[:, :columns]
    return df

# --- PARTSUPP TABLE ---
PARTSUPP_TABLE = "partsupp"
with conn.cursor() as cur:
    cur.execute(f"DROP TABLE IF EXISTS {PARTSUPP_TABLE};")
    cur.execute(f"""
        CREATE TABLE {PARTSUPP_TABLE} (
            ps_partkey      INTEGER NOT NULL,
            ps_suppkey      INTEGER NOT NULL,
            ps_availqty     INTEGER,
            ps_supplycost   DECIMAL,
            ps_comment      VARCHAR(199),
            ps_image_embedding vector(96),
            ps_text_embedding vector(768)
        );
    """)

partsupp_df = read_csv(partsupp_csv_path, 5)
num_rows = len(partsupp_df)
chunk_size = 10000
num_chunks = (num_rows + chunk_size - 1) // chunk_size

def process_partsupp_chunk(chunk_idx, chunk_size, image_embedding_file, text_embedding_file, partsupp_df):
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
            int(row[0]), int(row[1]), int(row[2]), Decimal(str(row[3])), str(row[4]),
            row[5], row[6]
        )
        for row in df_chunk.itertuples(index=False, name=None)
    ]

    with conn.cursor() as cur:
        copy_sql = f"""
            COPY {PARTSUPP_TABLE}
            (ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment, ps_image_embedding, ps_text_embedding)
            FROM STDIN WITH (FORMAT BINARY)
        """
        with cur.copy(copy_sql) as copy:
            copy.set_types(["int4", "int4", "int4", "numeric", "text", "vector", "vector"])
            for row in rows:
                copy.write_row(row)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(process_partsupp_chunk, i, chunk_size, deep_bin_path, wiki_bin_path, partsupp_df)
        for i in range(num_chunks)
    ]
    for fut in futures:
        fut.result()

# --- PART TABLE ---
PART_TABLE = "part"
with conn.cursor() as cur:
    cur.execute(f"DROP TABLE IF EXISTS {PART_TABLE};")
    cur.execute(f"""
        CREATE TABLE {PART_TABLE} (
            p_partkey      INTEGER NOT NULL,
            p_name         VARCHAR(55),
            p_mfgr         VARCHAR(25),
            p_brand        VARCHAR(10),
            p_type         VARCHAR(25),
            p_size         INTEGER,
            p_container    VARCHAR(10),
            p_retailprice  DECIMAL,
            p_comment      VARCHAR(23),
            text_embedding vector(768)
        );
    """)

part_df = read_csv(part_csv_path, 9)
num_part_rows = len(part_df)
wiki_vectors = read_fbin(wiki_bin_path, start_idx=0, chunk_size=num_part_rows)
part_df['text_embedding'] = list(wiki_vectors)

# Prepare rows for COPY
rows = [
    (
        int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]),
        int(row[5]), str(row[6]), Decimal(str(row[7])), str(row[8]), row[9]
    )
    for row in part_df.itertuples(index=False, name=None)
]

with conn.cursor() as cur:
    copy_sql = f"""
        COPY {PART_TABLE}
        (p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment, text_embedding)
        FROM STDIN WITH (FORMAT BINARY)
    """
    with cur.copy(copy_sql) as copy:
        copy.set_types(["int4", "text", "text", "text", "text", "int4", "text", "numeric", "text", "vector"])
        for row in rows:
            copy.write_row(row)

cur.execute("SET maintenance_work_mem = '8GB';")
cur.execute("SET max_parallel_maintenance_workers = 7; -- plus leader")

cur.execute("CREATE INDEX partsupp_deep_hnsw ON partsupp USING hnsw (ps_image_embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);")
cur.execute("CREATE INDEX partsupp_wiki_hnsw ON partsupp USING hnsw (ps_text_embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);")

cur.execute("CREATE INDEX part_wiki_hnsw ON part USING hnsw (text_embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);")

conn.close()

