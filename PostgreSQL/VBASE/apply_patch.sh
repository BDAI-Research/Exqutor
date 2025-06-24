#!/bin/bash

dir="$(cd "$(dirname "$0")" && pwd)"

cd "$dir" || exit 1

cd MSVBASE/thirdparty/Postgres
git apply ../../../patch/vbase_Postgres.patch
cd "$dir"

cd MSVBASE/thirdparty/hnsw
git apply ../../../patch/vbase_hnsw.patch
cd "$dir"


cd MSVBASE/thirdparty/SPTAG
git apply ../../../patch/vbase_spann.patch
cd "$dir"

cd MSVBASE
git apply ../patch/vbase_Exqutor.patch
git apply ../patch/vbase_Exqutor_lib.patch
cd "$dir"


