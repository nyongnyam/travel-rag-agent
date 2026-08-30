from crawler import build_vectorstore
from graph import build_app
import os

def main():
    if not os.path.exists("./chroma_db"):
        build_vectorstore(cities=["제주"], categories=["관광지", "음식점", "숙박"])

    app = build_app()

    user_request = input("여행 요청을 입력하세요 (예: 제주 2박3일 일정 짜줘): ")

    result = app.invoke({
        "user_request": user_request,
        "transport_info": "",
        "stay_info": "",
        "food_info": "",
        "final_itinerary": "",
    })
    print(result["final_itinerary"])

if __name__ == "__main__":
    main()