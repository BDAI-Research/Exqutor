#!/bin/bash

# 상대경로 설정
PATCH_FILE="PostgreSQL/pgvector/patch/pgvector_Exqutor.patch"
TARGET_DIR="PostgreSQL/pgvector/pgvector/postgres"

# 대상 디렉토리로 이동
cd "$TARGET_DIR" || { echo "Failed to enter $TARGET_DIR"; exit 1; }

# 패치 적용
patch -p1 < ../../patch/pgvector_Exqutor.patch

# 적용 결과 출력
if [ $? -eq 0 ]; then
  echo "Patch applied successfully."
else
  echo "Patch application failed. Please check for conflicts."
fi