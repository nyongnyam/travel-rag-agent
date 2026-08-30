![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Chroma](https://img.shields.io/badge/Chroma-FF6F61?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white)

# 공공데이터 기반 RAG 멀티에이전트 여행 일정 생성기

## 1. 프로젝트 개요

본 프로젝트는 한국관광공사가 공공데이터포털을 통해 제공하는 TourAPI 데이터를 대상으로 수집·정제·임베딩 파이프라인을 구축하고, 이를 기반으로 LangGraph 멀티에이전트가 사용자 요청에 맞는 여행 일정을 자동 생성하는 것을 목표로 한다.

### 1.1 프로젝트 목표

- 공공데이터포털 TourAPI를 자동으로 수집·정제해 벡터 데이터베이스에 적재하는 RAG 파이프라인 구축
- LangGraph 기반 Supervisor-Worker 멀티에이전트 아키텍처 설계 및 구현
- 로컬 경량 LLM(Qwen2.5-3B-Instruct)을 활용한 온디바이스 에이전트 서빙
- Gradio를 통한 웹 UI 구성

### 1.2 사용 데이터

**한국관광공사_국문 관광정보 서비스 (TourAPI)**

- 출처: 공공데이터포털(data.go.kr) — 한국관광공사_국문 관광정보 서비스_GW
- 형식: JSON
- 대상 카테고리: 관광지, 음식점, 숙박
- 주요 필드: 명칭, 주소, 개요(overview), 콘텐츠 ID, 지역 코드
- 수집 범위: 현재는 제주 지역으로 한정 (사유는 8절 한계 참고). API 구조상 지역 코드만 추가하면 전국으로 확장 가능하도록 설계됨

## 2. 시스템 아키텍처

- **데이터 수집**: Python `requests`를 통해 TourAPI에서 지역별·카테고리별 여행 정보를 JSON으로 수집
- **데이터 정제/청킹**: `langchain-text-splitters`의 `RecursiveCharacterTextSplitter`로 텍스트를 청크 단위로 분할
- **임베딩**: `BAAI/bge-m3` 다국어 임베딩 모델을 통해 청크를 벡터로 변환
- **벡터 적재**: `Chroma`를 통해 로컬 벡터 데이터베이스에 영구 저장
- **에이전트 오케스트레이션**: `LangGraph`의 `StateGraph`로 Supervisor-Worker 구조 구성
- **LLM 서빙**: HuggingFace `transformers` pipeline으로 `Qwen2.5-3B-Instruct`를 로컬에서 구동
- **UI**: `Gradio`로 웹 인터페이스 구성, 로컬 환경에서 실행
- **버전 관리**: Git을 통한 코드 관리, `.env` 기반 API 키 분리

## 3. RAG 파이프라인 상세

### 3.1 Extract (수집)

TourAPI의 `areaBasedList2` 엔드포인트를 통해 지역 코드(`areaCode`)와 카테고리 코드(`contentTypeId`) 조합으로 관광 정보를 수집한다.

- 수집 단위: 지역(시/도) × 카테고리(관광지/음식점/숙박)
- 페이지당 최대 100건, 카테고리별 순차 호출
- API 호출 간 `time.sleep()`으로 서버 부하 및 차단 방지
- 서비스키는 디코딩된 형태로 `.env`에 저장해 URL 이중 인코딩 문제 방지

### 3.2 Transform (정제/청킹)

**항목별 텍스트 변환**

| 원본 필드 | 변환 항목 | 비고 |
|---|---|---|
| `title` | 이름 | 장소명 |
| `addr1` | 주소 | 상세 주소 |
| `overview` | 설명 | 존재하는 경우만 포함 |

**청킹 기준**

| 파라미터 | 값 | 목적 |
|---|---|---|
| `chunk_size` | 500 | RAG 검색 단위 확보 |
| `chunk_overlap` | 50 | 문맥 단절 방지 |

**메타데이터 구성**

| 메타데이터 | 내용 | 활용 목적 |
|---|---|---|
| `city` | 지역명 | 지역 기반 필터링 |
| `category` | 관광지/음식점/숙박 | Worker별 검색 범위 제한 |
| `title` | 장소명 | 결과 표시 |
| `content_id` | TourAPI 콘텐츠 ID | 원본 데이터 추적 |

### 3.3 Load (적재)

정제된 청크를 `HuggingFaceEmbeddings(bge-m3)`로 임베딩한 뒤 `Chroma.from_documents()`를 통해 `chroma_db/` 디렉터리에 영구 저장한다. 최초 실행 이후에는 기존 벡터DB를 재사용해 크롤링을 반복하지 않는다.

## 4. 멀티에이전트 구조

### 4.1 Supervisor-Worker 설계

| 노드 | 역할 |
|---|---|
| `transport_worker` | 동선/교통 관련 정보를 RAG로 검색 후 요약 |
| `stay_worker` | 숙박 카테고리 필터링 검색 후 추천 숙소 정리 |
| `food_worker` | 음식점 카테고리 필터링 검색 후 추천 맛집 정리 |
| `supervisor` | 세 Worker의 결과를 종합해 일차별 최종 일정표 생성 |

### 4.2 상태(State) 정의

`TypedDict` 기반 `TravelState`로 사용자 요청과 Worker별 결과, 최종 일정을 하나의 상태 객체에서 관리한다.

### 4.3 실행 방식 및 트러블슈팅

초기에는 LangGraph의 fan-out/fan-in 구조로 세 Worker를 병렬 실행하도록 설계했으나, 단일 `transformers` 모델 인스턴스를 여러 스레드에서 동시 호출할 경우 디바이스 텐서 충돌(`RuntimeError: Tensor on device cpu is not on the expected device meta`)이 발생함을 확인했다. 이는 로컬 sLLM 인스턴스가 API 기반 모델과 달리 동시 호출에 대한 안전성을 보장하지 않기 때문으로, 순차 실행 구조(`transport → stay → food → supervisor`)로 전환해 안정성을 확보했다.

## 5. 실행 환경

### 5.1 프로젝트 구조

```
travel-rag-agent/
├── .env                  # TOUR_API_KEY 저장 (Git 제외)
├── .gitignore
├── requirements.txt
├── crawler.py            # TourAPI 수집 + 벡터스토어 적재
├── graph.py              # Supervisor-Worker LangGraph 정의
├── main.py               # CLI 실행 진입점
├── app.py                # Gradio UI 실행 진입점
└── chroma_db/            # 벡터DB 저장 폴더 (자동 생성)
```

### 5.2 설치 및 실행

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`.env` 파일에 공공데이터포털에서 발급받은 디코딩 서비스키를 입력한다.

```
TOUR_API_KEY=발급받은_디코딩_서비스키
```

**웹 UI 실행**
```bash
python app.py
```

**CLI 실행**
```bash
python main.py
```

## 6. 기술적 의사결정

- **로컬 sLLM 채택 이유**: API 호출 비용 없이 온디바이스에서 완결되는 에이전트 파이프라인을 구성하기 위해 경량 모델(Qwen2.5-3B-Instruct)을 채택
- **디코딩/인코딩 서비스키 이슈**: `requests` 사용 시 URL 인코딩된 키(Encoding 키)를 그대로 사용하면 자동 인코딩과 중복되어 인증 오류가 발생하므로, 반드시 디코딩된 키를 사용해야 함을 확인

- **로컬 리소스 제약에 따른 수집 범위 축소**: 전국 17개 지역 × 3개 카테고리(51회 API 호출) 데이터를 한 번에 임베딩하면서 로컬 LLM(Qwen2.5-3B) 로딩이 겹칠 경우, `device_map="auto"`의 디스크 오프로드 동작과 맞물려 메모리 과부하로 시스템이 강제 종료되는 현상을 확인함. 이에 따라 1차 구현 범위를 제주 지역으로 한정하고, `device_map="auto"` 대신 `device=-1`(CPU 명시)로 전환해 안정성을 확보함

## 7. 한계 및 아쉬운 점

프로젝트를 진행하면서 로컬 하드웨어 사양(CPU 환경, 16GB RAM) 안에서 감수할 수밖에 없었던 트레이드오프들이다.

- **데이터 수집 범위 한정**: 전국 데이터를 한 번에 적재하려다 메모리 부족으로 시스템이 다운되는 것을 겪은 뒤, 안정성을 우선해 제주 지역으로 범위를 좁힘. 코드 구조상 `AREA_CODES`에 정의된 지역을 추가하기만 하면 전국 확장이 가능하지만, 이를 실제로 검증하려면 더 넉넉한 메모리/GPU 환경이 필요함
- **프롬프트 튜닝 미흡**: Worker별 요약 프롬프트("2~3줄로 요약해" 등)가 단순한 수준에 그쳐, 생성되는 일정의 구체성(시간대, 장소별 설명 등)이 기대에 못 미치는 경우가 있었음. `max_new_tokens`을 늘리거나 출력 형식을 더 구체적으로 지시하는 프롬프트 엔지니어링을 시도했으나, CPU 환경에서는 토큰 수를 늘릴수록 응답 시간이 비례해 늘어나 충분히 반복 실험하지 못함
- **RAG 검색 파라미터(k값) 미세 조정 부족**: 카테고리별 검색 결과 개수(`k=5`)를 늘리면 답변 품질이 개선될 가능성이 있으나, 마찬가지로 CPU 환경에서의 응답 속도 문제로 다양한 값을 비교 실험하기 어려웠음
- **병렬 처리 포기**: Supervisor-Worker 구조를 설계한 목적 중 하나가 병렬 실행을 통한 응답 속도 개선이었으나, 단일 로컬 모델 인스턴스의 동시성 제약으로 순차 실행으로 전환하면서 이 이점을 살리지 못함

## 8. 향후 개선 방향

### 8.1 데이터 파이프라인

- 더 넉넉한 메모리 환경 또는 클라우드 리소스를 확보해 전국 단위로 수집 범위 확장
- 수동 실행 기반에서 스케줄러 연동을 통한 데이터 최신성 자동 확보로 전환

### 8.2 에이전트 품질

- GPU 또는 API 기반 모델 환경을 확보해 프롬프트/파라미터(k값, max_new_tokens 등) 튜닝을 반복 실험할 수 있는 환경 마련
- LLM-as-judge 기반 생성 일정 품질 평가(환각 장소 검출 등) 도입
- API 기반 모델 대체 시 Worker 간 진짜 병렬 실행 구조로 전환

### 8.3 실행 안정성

- CPU 환경에서의 메모리/발열 이슈를 고려한 모델 경량화(양자화, GGUF 등) 검토
- 로컬 환경에서의 안정적인 실행을 우선 목표로 하며, 외부 배포는 향후 과제로 남김
