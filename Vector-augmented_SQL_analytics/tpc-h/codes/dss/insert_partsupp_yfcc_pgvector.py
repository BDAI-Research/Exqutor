#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import numpy as np
import pandas as pd
import psycopg
import pgvector.psycopg
from rich import print

# =========================
# User settings (paths/parameters)
# =========================
PARTSUPP_CSV_PATH = '/mnt/hdd0/home/wns41559/exqutor/TPC-H/dbgen/partsupp.csv'
VECTOR_BIN_PATH   = '/mnt/hdd0/home/wns41559/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin'
LABELS_JSON_PATH  = '/home/wns41559/pg/yfcc/labels.json'  # Tag JSON path (e.g., [[6,15,...],[7,17,...],...])

# Whether to normalize u8bin to float32 (0~255 → 0~1)
NORMALIZE_U8 = False

# Table/index/vector dimension
TABLE_NAME = 'partsupp_yfcc_10'
VEC_DIM = 192

# DB connection settings
PG_HOST = '/tmp'
PG_PORT = 5400

# Loading settings
CHUNK_SIZE = 10_000
MAX_WORKERS = 32

# Index creation (after loading)
CREATE_INDEXES = True
# HNSW parameters (tune if needed)
HNSW_OPS = 'vector_l2_ops'
HNSW_WITH = "WITH (m=16, ef_construction=200)"


# =========================
# DB connection/table preparation
# =========================
def create_connection():
    conn = psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
    )
    pgvector.psycopg.register_vector(conn)
    conn.autocommit = True
    return conn


def drop_and_create_table():
    with create_connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                ps_partkey    BIGINT NOT NULL,
                ps_suppkey    BIGINT NOT NULL,
                ps_availqty   INTEGER,
                ps_supplycost DECIMAL,
                ps_comment    TEXT,
                ps_embedding  vector({VEC_DIM}),
                ps_tags       INT[]
            );
        """)
    print(f"[bold green]Table {TABLE_NAME} created successfully![/bold green]")


# =========================
# File loaders
# =========================
def read_partsupp(partsupp_file_path: str) -> pd.DataFrame:
    start = time.time()
    ps = pd.read_csv(partsupp_file_path, delimiter='|', header=None, engine="pyarrow")
    end = time.time()
    print(f"Reading CSV file took {end - start:.2f} seconds")
    return ps


def read_json(path: str):
    """Load JSON file ([[6,15,...], [7,17,...], ...]) as a list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def normalize_tags_length(tags_list, target_len):
    """
    Adjust tags_list length to match the number of CSV rows (target_len).
    - Truncate if too long
    - Pad with empty lists if too short
    Also, force each element to be list[int].
    """
    if len(tags_list) > target_len:
        tags_list = tags_list[:target_len]
        print(f"[bold yellow]Truncated tags_list to {target_len} rows[/bold yellow]")
    elif len(tags_list) < target_len:
        pad = target_len - len(tags_list)
        tags_list = list(tags_list) + ([[]] * pad)
        print(f"[bold yellow]Padded tags_list with {pad} empty entries to match CSV rows[/bold yellow]")

    # Normalize each element to list[int]
    tags_list = [list(map(int, t)) if isinstance(t, (list, tuple)) else [] for t in tags_list]
    return tags_list


def read_fbin(filename: str, start_idx=0, chunk_size=None) -> np.ndarray:
    """Read FLOAT32 .fbin (nvecs, dim, data...) format"""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        n_fetch = (nvecs - start_idx) if chunk_size is None else chunk_size
        offset_bytes = start_idx * 4 * dim  # float32=4 bytes
        arr = np.fromfile(f, count=n_fetch * dim, dtype=np.float32, offset=offset_bytes)
        if len(arr) == 0 or len(arr) % dim != 0:
            print(f"[bold red]Error:[/bold red] Invalid vector data at start_idx={start_idx}, chunk_size={chunk_size}")
            return np.empty((0, dim), dtype=np.float32)
    return arr.reshape(-1, dim)


def read_ubin(filename: str, start_idx=0, chunk_size=None) -> np.ndarray:
    """
    Read U8BIN (nvecs, dim, data...) format and return as float32.
    Default: normalize 0~255 → 0~1 (if NORMALIZE_U8=True)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        n_fetch = (nvecs - start_idx) if chunk_size is None else chunk_size
        offset_bytes = start_idx * 1 * dim  # uint8=1 byte
        arr = np.fromfile(f, count=n_fetch * dim, dtype=np.uint8, offset=offset_bytes)
        if len(arr) == 0 or len(arr) % dim != 0:
            print(f"[bold red]Error:[/bold red] Invalid vector data at start_idx={start_idx}, chunk_size={chunk_size}")
            return np.empty((0, dim), dtype=np.float32)
    arr = arr.reshape(-1, dim).astype(np.float32)
    if NORMALIZE_U8:
        arr = arr / 255.0
    return arr


# =========================
# Chunk processing (COPY)
# =========================
def process_chunk(chunk_idx: int, chunk_size: int, embedding_file_path: str, partsupp_df: pd.DataFrame, tags_list):
    start_index = chunk_idx * chunk_size
    end_index = min(start_index + chunk_size, len(partsupp_df))
    print(f"[Thread {threading.get_ident()}] [bold yellow]Processing chunk {chunk_idx}[/bold yellow] - [{start_index} to {end_index}]")

    try:
        # Read vectors (using u8bin)
        vectors = read_ubin(embedding_file_path, start_idx=start_index, chunk_size=(end_index - start_index))

        expected = end_index - start_index
        if len(vectors) != expected:
            print(f"[bold red]Warning:[/bold red] Vector length mismatch at chunk {chunk_idx}. "
                  f"Expected {expected}, got {len(vectors)}")
            return

        # Slice partsupp
        partsupp_df_chunk = partsupp_df.iloc[start_index:end_index].copy()
        partsupp_df_chunk['ps_embedding'] = vectors.tolist()

        # Slice tags (using already normalized tags_list)
        tags_chunk = tags_list[start_index:end_index]
        if len(tags_chunk) != expected:
            print(f"[bold red]Warning:[/bold red] Tags length mismatch at chunk {chunk_idx}. "
                  f"Expected {expected}, got {len(tags_chunk)}")
            return

        partsupp_df_chunk['ps_tags'] = tags_chunk

        # COPY
        with create_connection() as conn, conn.cursor() as cur:
            copy_sql = f"""
                COPY {TABLE_NAME}
                (ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment, ps_embedding, ps_tags)
                FROM STDIN WITH (FORMAT BINARY)
            """
            with cur.copy(copy_sql) as copy:
                # Column types: int8, int8, int4, numeric, text, vector, int4[]
                copy.set_types(["int8", "int8", "int4", "numeric", "text", "vector", "int4[]"])

                for row in partsupp_df_chunk.itertuples(index=False, name=None):
                    prepared_row = (
                        int(row[0]),             # ps_partkey
                        int(row[1]),             # ps_suppkey
                        int(row[2]),             # ps_availqty
                        Decimal(str(row[3])),    # ps_supplycost
                        str(row[4]),             # ps_comment
                        row[5],                  # ps_embedding (list[float])
                        row[6],                  # ps_tags (list[int])
                    )
                    try:
                        copy.write_row(prepared_row)
                    except Exception as e:
                        print(f"[bold red]Row insertion failed at chunk {chunk_idx}:[/bold red] {row}")
                        print(f"[bold red]Error message:[/bold red] {str(e)}")

    except Exception as e:
        print(f"[Thread {threading.get_ident()}] [bold red]Error processing chunk {chunk_idx}:[/bold red] {str(e)}")


# =========================
# Index creation
# =========================
def create_indexes():
    with create_connection() as conn, conn.cursor() as cur:
        print("[bold blue]Creating GIN index on ps_tags...[/bold blue]")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tags_gin
            ON {TABLE_NAME} USING GIN (ps_tags);
        """)

        print("[bold blue]Creating HNSW index on ps_embedding...[/bold blue]")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_emb_hnsw
            ON {TABLE_NAME} USING hnsw (ps_embedding {HNSW_OPS})
            {HNSW_WITH};
        """)
    print("[bold green]Indexes created![/bold green]")


# =========================
# Main
# =========================
def main():
    # Create table
    drop_and_create_table()

    # Read PARTSUPP
    ps = read_partsupp(PARTSUPP_CSV_PATH)
    print(f"Data shape: {ps.shape}")

    # Read tag JSON + normalize length
    raw_tags = read_json(LABELS_JSON_PATH)
    tags_list = normalize_tags_length(raw_tags, len(ps))   # ★ Normalize length to match CSV

    # Calculate chunks
    num_rows = len(ps)
    chunk_size = CHUNK_SIZE
    num_chunks = (num_rows + chunk_size - 1) // chunk_size
    start_chunk = 0
    end_chunk = num_chunks
    print(f"Number of chunks: {num_chunks}")

    # Multithreaded loading
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_chunk, i, chunk_size, VECTOR_BIN_PATH, ps, tags_list)
            for i in range(start_chunk, end_chunk)
        ]
        for fut in futures:
            fut.result()
    t1 = time.time()
    print(f"[bold green]Data insertion completed successfully from {start_chunk} to {end_chunk}![/bold green]")
    print(f"[bold cyan]Elapsed: {(t1 - t0):.2f} sec[/bold cyan]")

    # Create indexes
    if CREATE_INDEXES:
        create_indexes()


if __name__ == "__main__":
    main()