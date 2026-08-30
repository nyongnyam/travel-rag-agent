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


def rag_search(query: str, category: str, k: int = 5) -> str:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"category": category}}
    )
    results = retriever.invoke(query)
    if not results:
        return "관련 정보 없음"
    return "\n".join(f"- {d.page_content}" for d in results)


def transport_worker(state: TravelState) -> dict:
    info = rag_search(state["user_request"], category="관광지")
    prompt = f"다음 정보를 바탕으로 이동/동선 관련 팁을 2~3줄로 요약해:\n{info}"
    result = llm.invoke(prompt)
    return {"transport_info": result.content}


def stay_worker(state: TravelState) -> dict:
    info = rag_search(state["user_request"], category="숙박")
    prompt = f"다음 숙소 정보를 바탕으로 추천 숙소 2곳을 이유와 함께 정리해:\n{info}"
    result = llm.invoke(prompt)
    return {"stay_info": result.content}


def food_worker(state: TravelState) -> dict:
    info = rag_search(state["user_request"], category="음식점")
    prompt = f"다음 음식점 정보를 바탕으로 추천 맛집 3곳을 정리해:\n{info}"
    result = llm.invoke(prompt)
    return {"food_info": result.content}


def supervisor_synthesize(state: TravelState) -> dict:
    prompt = f"""너는 여행 일정 플래너야. 아래 정보를 종합해서 일차별(Day1, Day2...) 일정표를 만들어.

[사용자 요청]
{state['user_request']}

[동선/교통 정보]
{state['transport_info']}

[숙소 정보]
{state['stay_info']}

[맛집 정보]
{state['food_info']}

일차별, 시간대별로 정리해서 답해."""
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