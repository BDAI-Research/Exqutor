#!/bin/bash

dir="$(cd "$(dirname "$0")" && pwd)"

cd "$dir" || exit 1

if [ -d MSVBASE ]; then
  cd MSVBASE
  ./scripts/patch.sh
  git apply ../patch/vbase_Exqutor.patch || true
  cd "$dir"
fi

if [ -d MSVBASE/thirdparty/Postgres ]; then
  cd MSVBASE/thirdparty/Postgres
  git apply ../../../../patch/vbase_Postgres.patch || true
  cd "$dir"
fi