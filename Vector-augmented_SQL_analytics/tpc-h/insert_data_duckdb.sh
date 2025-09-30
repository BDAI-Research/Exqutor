#!/bin/bash

dir=$(dirname "$0")

cd "$dir"/codes/dss

python insert_data_duckdb.py
psql tpch -f tpch-load.sql
psql tpch -f tpch-alter.sql
psql tpch -f tpch-index.sql

