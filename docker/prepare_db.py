"""컨테이너 시작 시 벡터DB를 점검하고, DATA_CITIES 중 아직 적재되지 않은 지역만 수집해 추가한다.

- main.py는 ./chroma_db 폴더 존재 여부만 보고 크롤링을 건너뛰기 때문에,
  빈 폴더가 있으면 RAG 검색 결과가 항상 '관련 정보 없음'이 된다.
  여기서는 실제로 적재된 지역을 메타데이터로 확인해 그 문제를 피한다.
- 이미 적재된 지역은 다시 수집하지 않으므로, DATA_CITIES에 지역을 추가하고
  재시작하면 새 지역만 덧붙여진다.
"""
import os
import sys

sys.path.insert(0, "/app")

PERSIST_DIR = "./chroma_db"
CATEGORIES = ["관광지", "음식점", "숙박"]


def requested_cities() -> list[str]:
    from crawler import AREA_CODES
    cities = [c.strip() for c in os.getenv("DATA_CITIES", "제주").split(",") if c.strip()]
    unknown = [c for c in cities if c not in AREA_CODES]
    if unknown:
        sys.exit(f"[prepare_db] 알 수 없는 지역명: {unknown}\n사용 가능: {', '.join(AREA_CODES)}")
    return cities


def loaded_cities() -> set[str]:
    if not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        return set()
    import chromadb
    try:
        col = chromadb.PersistentClient(path=PERSIST_DIR).get_collection("langchain")
        metas = col.get(include=["metadatas"])["metadatas"] or []
    except Exception:
        return set()
    return {m.get("city") for m in metas if m and m.get("city")}


def main():
    cities = requested_cities()
    loaded = loaded_cities()
    missing = [c for c in cities if c not in loaded]
    if not missing:
        print(f"[prepare_db] 기존 벡터DB 사용 (적재된 지역: {', '.join(sorted(loaded))})")
        return
    if not os.getenv("TOUR_API_KEY"):
        sys.exit("[prepare_db] 새로 적재할 지역이 있는데 TOUR_API_KEY가 없습니다. .env를 확인하세요.")
    print(f"[prepare_db] 새로 적재할 지역: {missing} (이미 있음: {sorted(loaded) or '없음'})")
    from crawler import build_vectorstore
    build_vectorstore(cities=missing, categories=CATEGORIES, persist_dir=PERSIST_DIR)


if __name__ == "__main__":
    main()
