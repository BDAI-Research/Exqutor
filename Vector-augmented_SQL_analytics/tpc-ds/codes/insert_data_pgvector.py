import pandas as pd
import numpy as np
import psycopg
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

# File paths
deep_bin_path = '../../dataset/DEEP/base.1B.fbin'
item_csv_path = '../../dataset/third_party/tpcds-kit/tools/item.csv'

# DB connection
conn = psycopg.connect(dbname='tpcds')
conn.autocommit = True
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vectordb;")

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

# --- ITEM TABLE ---
ITEM_TABLE = "item"
cur.execute(f"DROP TABLE IF EXISTS {ITEM_TABLE};")
cur.execute(f"""
    CREATE TABLE {ITEM_TABLE} (
        i_itemkey      INTEGER NOT NULL,
        i_name         VARCHAR(55),
        i_mfgr         VARCHAR(25),
        i_brand        VARCHAR(10),
        i_type         VARCHAR(25),
        i_size         INTEGER,
        i_container    VARCHAR(10),
        i_retailprice  DECIMAL,
        i_comment      VARCHAR(23),
        image_embedding vector(96)
    );
""")

item_df = read_csv(item_csv_path, 9)
num_item_rows = len(item_df)
deep_vectors = read_fbin(deep_bin_path, start_idx=0, chunk_size=num_item_rows)
item_df['image_embedding'] = list(deep_vectors)

# Prepare rows for COPY
rows = [
    (
        int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]),
        int(row[5]), str(row[6]), Decimal(str(row[7])), str(row[8]), row[9]
    )
    for row in item_df.itertuples(index=False, name=None)
]

copy_sql = f"""
    COPY {ITEM_TABLE}
    (i_itemkey, i_name, i_mfgr, i_brand, i_type, i_size, i_container, i_retailprice, i_comment, image_embedding)
    FROM STDIN WITH (FORMAT BINARY)
"""
with cur.copy(copy_sql) as copy:
    copy.set_types(["int4", "text", "text", "text", "text", "int4", "text", "numeric", "text", "vector"])
    for row in rows:
        copy.write_row(row)

cur.execute("SET maintenance_work_mem = '8GB';")
cur.execute("SET max_parallel_maintenance_workers = 7; -- plus leader")
cur.execute(f"CREATE INDEX item_deep_hnsw ON {ITEM_TABLE} USING hnsw (image_embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);")

conn.close()