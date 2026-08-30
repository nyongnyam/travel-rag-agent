import os, time
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
SERVICE_KEY = os.getenv("TOUR_API_KEY")
BASE_URL = "https://apis.data.go.kr/B551011/KorService2"

AREA_CODES = {
    "서울": 1, "인천": 2, "대전": 3, "대구": 4, "광주": 5,
    "부산": 6, "울산": 7, "세종": 8, "경기": 31, "강원": 32,
    "충북": 33, "충남": 34, "경북": 35, "경남": 36, "전북": 37,
    "전남": 38, "제주": 39,
}
CONTENT_TYPES = {
    "관광지": 12, "문화시설": 14, "축제공연행사": 15,
    "여행코스": 25, "레포츠": 28, "숙박": 32, "쇼핑": 38, "음식점": 39,
}


def fetch_area_list(area_name: str, content_type: str, num_rows: int = 100) -> list[dict]:
    params = {
        "serviceKey": SERVICE_KEY,
        "areaCode": AREA_CODES[area_name],
        "contentTypeId": CONTENT_TYPES[content_type],
        "numOfRows": num_rows,
        "pageNo": 1,
        "MobileOS": "ETC",
        "MobileApp": "trip_rag",
        "_type": "json",
        "arrange": "A",
    }
    res = requests.get(f"{BASE_URL}/areaBasedList2", params=params, timeout=10)
    res.raise_for_status()
    body = res.json()["response"]["body"]
    if body["totalCount"] == 0:
        return []
    items = body["items"]["item"]
    return items if isinstance(items, list) else [items]


def item_to_text(item: dict) -> str:
    parts = [
        f"이름: {item.get('title', '')}",
        f"주소: {item.get('addr1', '')}",
        f"설명: {item.get('overview', '')}" if item.get("overview") else "",
    ]
    return "\n".join(p for p in parts if p)


def collect_documents(area_name: str, content_type: str) -> list[Document]:
    items = fetch_area_list(area_name, content_type)
    docs = []
    for item in items:
        docs.append(Document(
            page_content=item_to_text(item),
            metadata={
                "city": area_name,
                "category": content_type,
                "title": item.get("title", ""),
                "content_id": item.get("contentid", ""),
            }
        ))
    return docs


def build_vectorstore(cities: list[str], categories: list[str], persist_dir: str = "./chroma_db"):
    all_docs = []
    for city in cities:
        for category in categories:
            docs = collect_documents(city, category)
            all_docs.extend(docs)
            print(f"{city}-{category}: {len(docs)}건 수집")
            time.sleep(0.3)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
    print(f"총 {len(split_docs)}개 청크 적재 완료")
    return vectorstore