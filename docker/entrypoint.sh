#!/bin/sh
set -e
# 벡터DB가 비어 있으면 TourAPI 수집 + 임베딩 적재를 먼저 수행
python /app/docker/prepare_db.py
exec "$@"
