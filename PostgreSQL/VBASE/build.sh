#!/bin/bash

dir="$(cd "$(dirname "$0")" && pwd)"

export PGBASE="$(pwd)/psql"

cd MSVBASE/thirdparty/Postgres || exit 1

./configure \
  --with-blocksize=32 \
  --enable-integer-datetimes \
  --enable-thread-safety \
  --with-pgport=5432 \
  --prefix=${PGBASE} \
  --with-ldap \
  --with-python \
  --with-openssl \
  --with-libxml \
  --with-libxslt \
  --enable-nls=yes

make -j
make install
make -C contrib install

cd "${dir}" || exit 1
cd MSVBASE || exit 1

mkdir build && cd build

cmake -DCMAKE_INSTALL_PREFIX=${PGBASE}/share/postgresql -DLIBRARYONLY=ON -DSEEK_ENABLE_TESTS=ON \
-DPostgreSQL_INCLUDE_DIR=${PGBASE}/include/postgresql/server \
-DPostgreSQL_LIBRARY=${PGBASE}/lib/libpq.so \
-DPostgreSQL_TYPE_INCLUDE_DIR=${PGBASE}/include \
-DCMAKE_BUILD_TYPE=Release .. && \
make -j$(nproc)

cd ..

cp ./build/vectordb.so ${PGBASE}/lib/postgresql/vectordb.so
cp ./build/vectordb.control ${PGBASE}/share/postgresql/extension/vectordb.control
cp ./sql/vectordb.sql ${PGBASE}/share/postgresql/extension/vectordb--0.1.0.sql
