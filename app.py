import gradio as gr
from graph import build_app

travel_app = build_app()


def generate_itinerary(destination: str, days: int):
    if not destination.strip():
        return "여행지를 입력해줘!"

    user_request = f"{destination} {days}일 일정 짜줘"

    result = travel_app.invoke({
        "user_request": user_request,
        "transport_info": "",
        "stay_info": "",
        "food_info": "",
        "final_itinerary": "",
    })
    return result["final_itinerary"]


with gr.Blocks(title="여행 일정 생성기") as demo:
    gr.Markdown("# 🧳 RAG 기반 여행 일정 생성기")
    gr.Markdown("여행지와 기간을 입력하면 AI가 교통/숙소/맛집을 종합해 일정표를 만들어줘요.")

    with gr.Row():
        destination_input = gr.Textbox(label="여행지", placeholder="예: 제주도, 부산, 서울")
        days_input = gr.Number(label="기간(일)", value=3, precision=0)

    generate_btn = gr.Button("일정 생성하기", variant="primary")
    output_box = gr.Textbox(label="생성된 일정", lines=20)

    generate_btn.click(
        fn=generate_itinerary,
        inputs=[destination_input, days_input],
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch()