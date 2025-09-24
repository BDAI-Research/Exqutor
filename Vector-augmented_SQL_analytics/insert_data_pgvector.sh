#!/bin/bash

dir=$(dirname "$0")

cd "$dir"/tpc-h/codes

python insert_data_pgvector.py
cd dss
psql tpch -f tpch-load.sql
psql tpch -f tpch-alter.sql
psql tpch -f tpch-index.sql

