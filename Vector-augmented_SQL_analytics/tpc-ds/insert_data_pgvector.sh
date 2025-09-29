#!/bin/bash

dir=$(dirname "$0")

cd "$dir"/codes

python insert_data_pgvector.py
psql tpch -f load.sql

