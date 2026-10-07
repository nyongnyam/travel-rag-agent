# CPU 전용 이미지 (README의 실행 환경과 동일하게 device=-1, float32 기준)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace 모델(bge-m3, Qwen2.5-3B) 캐시 위치 → 볼륨으로 마운트해 재다운로드 방지
    HF_HOME=/cache/huggingface \
    # 컨테이너 밖에서 Gradio에 접속할 수 있도록 0.0.0.0 바인딩
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# torch는 CPU 휠을 먼저 설치 (기본 PyPI 휠은 CUDA 포함으로 수 GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY docker/ ./docker/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
