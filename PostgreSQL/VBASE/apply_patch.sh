#!/bin/bash

[ -d MSVBASE ] || exit 1
cd MSVBASE || exit 1
patch -p1 < ../patch/vbase_Exqutor.patch || exit 1
cd - > /dev/null

[ -d MSVBASE/thirdparty/Postgres ] || exit 1
cd MSVBASE/thirdparty/Postgres || exit 1
patch -p1 < ../../../../patch/vbase_Postgres.patch || exit 1
cd - > /dev/null