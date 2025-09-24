#!/bin/bash
dir=$(dirname "$0")
cd "$dir" || exit 1
./apply_patch.sh

psql="$(pwd)/psql"

cd postgres
./configure --prefix="$psql"
make
make install
cd contrib
make
make install

cd "$dir"/pgvector
export PG_CONFIG="$psql/bin/pg_config"
make 
make install

cd "$dir"/pg_hint_plan
export PG_CONFIG="$psql/bin/pg_config"
make
make install

cd "$psql/bin"
./pg_ctl -D "../data" -l ../log.log start
./createdb tpch

./psql tpch -c "CREATE EXTENSION vector;"

./createdb tpcds
./psql tpcds -c "CREATE EXTENSION vector;"