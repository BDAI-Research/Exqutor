#!/bin/bash

prefix="$(pwd)/psql"

cd postgres
./configure --prefix="$prefix"
make
make install

cd ../pgvector
export PG_CONFIG="$prefix/bin/pg_config"
make 
make install