#!/bin/bash

dir="$(cd "$(dirname "$0")" && pwd)"

cd "$dir" || exit 1

cd MSVBASE/thirdparty/Postgres
git apply ../../../patch/vbase_Postgres.patch || true
cd "$dir"

cd MSVBASE/thirdparty/hnsw
git apply ../../../patch/vbase_hnsw.patch || true
cd "$dir"


cd MSVBASE/thirdparty/spann
git apply ../../../patch/vbase_spann.patch || true
cd "$dir"

cd MSVBASE
git apply ../patch/vbase_Exqutor.patch || true
cd "$dir"


