#!/bin/bash

if [ -d "MSVBASE" ]; then
  cd MSVBASE || { echo "Failed to enter MSVBASE"; exit 1; }
  patch -p1 < ../patch/vbase_Exqutor.patch
  if [ $? -eq 0 ]; then
    echo "vbase_Exqutor.patch applied successfully in MSVBASE."
  else
    echo "Failed to apply vbase_Exqutor.patch in MSVBASE."
    exit 1
  fi
  cd - > /dev/null
else
  echo "MSVBASE directory not found."
  exit 1
fi

if [ -d "MSVBASE/thirdparty/Postgres" ]; then
  cd MSVBASE/thirdparty/Postgres || { echo "Failed to enter MSVBASE/thirdparty/Postgres"; exit 1; }
  patch -p1 < ../../../../patch/vbase_Postgres.patch
  if [ $? -eq 0 ]; then
    echo "vbase_Postgres.patch applied successfully in MSVBASE/thirdparty/Postgres."
  else
    echo "Failed to apply vbase_Postgres.patch in MSVBASE/thirdparty/Postgres."
    exit 1
  fi
  cd - > /dev/null
else
  echo "MSVBASE/thirdparty/Postgres directory not found."
  exit 1
fi