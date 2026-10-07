from typing import TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langgraph.graph import StateGraph, END, START
import transformers, torch


class TravelState(TypedDict):
    user_request: str
    transport_info: str
    stay_info: str
    food_info: str
    final_itinerary: str


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

pipe = transformers.pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    dtype=torch.float32,      
    device=-1,                
    max_new_tokens=512,
    return_full_text=False,
)
llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))


# 검색 결과 1건당 최대 글자 수 (긴 overview가 프롬프트를 불리는 것 방지)
MAX_CHARS_PER_DOC = 200


def rag_search(query: str, category: str, k: int = 3) -> str:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"category": category}}
    )
    results = retriever.invoke(query)
    if not results:
        return "관련 정보 없음"
    lines = []
    for d in results:
        text = " / ".join(d.page_content.split("\n"))
        if len(text) > MAX_CHARS_PER_DOC:
            text = text[:MAX_CHARS_PER_DOC] + "…"
        lines.append(f"- {text}")
    return "\n".join(lines)


# Worker는 카테고리별 RAG 검색만 담당하고 LLM을 호출하지 않는다.
# CPU 환경에서 시간이 가장 많이 드는 건 토큰 생성(decode)이므로,
# LLM 호출을 supervisor 1회로 모아 전체 응답 시간을 줄인다.
def transport_worker(state: TravelState) -> dict:
    return {"transport_info": rag_search(state["user_request"], category="관광지")}


def stay_worker(state: TravelState) -> dict:
    return {"stay_info": rag_search(state["user_request"], category="숙박")}


def food_worker(state: TravelState) -> dict:
    return {"food_info": rag_search(state["user_request"], category="음식점")}


def supervisor_synthesize(state: TravelState) -> dict:
    prompt = f"""너는 여행 일정 플래너야. 아래 검색 결과에 있는 장소만 사용해서 일차별(Day1, Day2...) 일정표를 만들어.

[사용자 요청]
{state['user_request']}

[관광지 후보]
{state['transport_info']}

[숙소 후보]
{state['stay_info']}

[음식점 후보]
{state['food_info']}

규칙:
- 일차별로 오전/점심/오후/저녁/숙소 순서로 정리해.
- 목록에 없는 장소는 만들어내지 마.
- 같은 날 방문하는 장소는 주소가 가까운 곳끼리 묶어."""
    result = llm.invoke(prompt)
    return {"final_itinerary": result.content}


def build_app():
    graph = StateGraph(TravelState)
    graph.add_node("transport_worker", transport_worker)
    graph.add_node("stay_worker", stay_worker)
    graph.add_node("food_worker", food_worker)
    graph.add_node("supervisor", supervisor_synthesize)

    graph.set_entry_point("transport_worker")
    graph.add_edge("transport_worker", "stay_worker")
    graph.add_edge("stay_worker", "food_worker")
    graph.add_edge("food_worker", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile()