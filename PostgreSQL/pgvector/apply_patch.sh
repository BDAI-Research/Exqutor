#!/bin/bash

# Move to script directory (pgvector/)
cd "$(dirname "$0")" || exit 1

# Apply pgvector_Exqutor.patch to ./pgvector
cd pgvector || exit 1
git checkout v0.7.1
patch -p1 < ../patch/pgvector_Exqutor.patch || exit 1
cd ..

# Apply pgvector_Postgres.patch to ./postgres
cd postgres || exit 1
patch -p1 < ../patch/pgvector_Postgres.patch || exit 1