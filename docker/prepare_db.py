"""컨테이너 시작 시 chroma_db에 데이터가 있는지 확인하고, 비어 있으면 적재한다.

main.py는 ./chroma_db 폴더 존재 여부만 보고 크롤링을 건너뛰기 때문에,
빈 폴더가 있으면 RAG 검색 결과가 항상 '관련 정보 없음'이 된다.
여기서는 실제 임베딩 개수를 확인해 그 문제를 피한다.
"""
import os
import sys

sys.path.insert(0, "/app")

PERSIST_DIR = "./chroma_db"
CITIES = [c.strip() for c in os.getenv("DATA_CITIES", "제주").split(",") if c.strip()]
CATEGORIES = ["관광지", "음식점", "숙박"]


def existing_count() -> int:
    if not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        return 0
    import chromadb
    try:
        return chromadb.PersistentClient(path=PERSIST_DIR).get_collection("langchain").count()
    except Exception:
        return 0


def main():
    n = existing_count()
    if n > 0:
        print(f"[prepare_db] 기존 벡터DB 사용 ({n}개 청크)")
        return
    if not os.getenv("TOUR_API_KEY"):
        sys.exit("[prepare_db] 벡터DB가 비어 있는데 TOUR_API_KEY가 없습니다. .env를 확인하세요.")
    print(f"[prepare_db] 벡터DB가 비어 있어 수집을 시작합니다: {CITIES}")
    from crawler import build_vectorstore
    build_vectorstore(cities=CITIES, categories=CATEGORIES, persist_dir=PERSIST_DIR)


if __name__ == "__main__":
    main()
