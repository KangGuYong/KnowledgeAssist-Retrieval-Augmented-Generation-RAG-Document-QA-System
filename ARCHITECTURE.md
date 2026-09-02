# 기술 문서 (Architecture)

이 문서는 **실제 코드에 구현된 내용**을 기준으로 작성되었다. 루트 `README.md`의
기술 스택 절은 초기 스캐폴딩 시점의 것으로, Anthropic/OpenAI API·MiniLM 임베딩 등
지금은 쓰이지 않는 구성을 설명하고 있으므로 이 문서를 우선한다.

핵심 성격은 **완전 로컬 실행형 한국어 RAG**다. 외부 API 키가 필요 없고, LLM·임베딩·
PDF 파싱·OCR이 전부 자체 호스팅된다.

---

## 1. 시스템 구성

```
┌──────────────┐        ┌───────────────────────────────────────────┐
│  React SPA   │  HTTP  │            FastAPI (:8000)                │
│  (Vite:5173) ├───────▶│  /upload  /chat  /documents               │
└──────────────┘        └───┬─────────────┬─────────────┬───────────┘
                            │             │             │
                  ┌─────────▼───┐  ┌──────▼──────┐  ┌───▼──────────┐
                  │   MinerU    │  │  PaddleOCR  │  │   Ollama     │
                  │  (:8100)    │  │ (별도 프로세스)│  │  (원격 :11434)│
                  │ PDF 레이아웃  │  │  이미지→텍스트 │  │     LLM      │
                  └─────────────┘  └─────────────┘  └──────────────┘
                            │
                  ┌─────────▼──────────────────────────┐
                  │  KURE-v1 임베딩 (GPU) → ChromaDB    │
                  └────────────────────────────────────┘
```

프로세스는 3~4개가 독립적으로 뜬다: FastAPI 백엔드, Vite 개발 서버, MinerU 서비스,
그리고 Ollama(원격 호스트에 있어도 무방). PaddleOCR만 백엔드가 직접 자식 프로세스로
띄운다(§4.2 참조).

---

## 2. 기술 스택

### 백엔드 (`backend/requirements.txt`)


| 영역               | 채택 기술                                  | 버전             | 비고                           |
| -------------------- | -------------------------------------------- | ------------------ | -------------------------------- |
| 웹 프레임워크      | FastAPI + Uvicorn                          | 0.109.0 / 0.27.0 |                                |
| 설정·검증         | Pydantic / pydantic-settings               | 2.13.5 / 2.15.0  | `.env` 로딩                    |
| RAG 오케스트레이션 | LangChain / langchain-community            | 1.3.18 / 0.4.2   | 현행 1.x 라인                  |
| LLM 클라이언트     | langchain-ollama                           | 1.1.0            | community에서 이관됨           |
| 벡터 DB            | ChromaDB                                   | 0.4.22           | 로컬 영속                      |
| 임베딩             | sentence-transformers + `nlpai-lab/KURE-v1` | 3.3.1            | 한국어 특화                    |
| LLM                | Ollama (`gemma4:26b-a4b-it-q4_K_M`)        | —               | HTTP 원격 호출                 |
| PDF 파싱           | MinerU (HTTP 서비스)                       | —               | 레이아웃·표·수식             |
| OCR                | PaddleOCR + PaddlePaddle                   | 3.7.0 / 3.2.2    | 한국어 인식 모델               |
| PDF 폴백           | pypdf / PyMuPDF                            | 4.0.0 / 1.28.2   | MinerU 실패 시                 |
| 테스트             | pytest                                     | 7.4.4            | 175개                          |

`langchain-experimental`은 **의존성에 없다.** 시멘틱 청킹을 직접 포팅했기 때문이다(§4.4).

`langchain-classic`은 **우리 코드가 import하지 않는다.** LangChain 1.x가 본체에서
제거한 레거시 체인을 담고 있어 한때 발판으로 썼으나, LCEL 재작성으로 걷어냈다.
`langchain-community`가 계속 끌고 오는 전이 패키지로만 남는다. 배경은
[LangChain 1.x 마이그레이션](docs/superpowers/specs/2026-09-02-langchain-1x-migration-design.md)과
[LCEL 재작성](docs/superpowers/specs/2026-09-02-lcel-rewrite-design.md).

### 프론트엔드 (`frontend/package.json`)

React 18.2 / TypeScript 5.3 / Vite 5.0 / axios 1.6 / react-dropzone 14.2 /
react-markdown 9.0 / DOMPurify 3.4 / lucide-react.

상태 관리 라이브러리는 쓰지 않는다. `App.tsx`가 문서 목록과 선택 상태를 소유하고
하위 컴포넌트에 props로 내려주는 단일 소유 구조다.

---

## 3. 데이터 흐름

### 3.1 문서 수집 (Ingestion)

```
PDF 업로드
   │
   ├─▶ 확장자·크기·문서 수 상한 검증          upload.py
   ├─▶ app/storage/uploads/{doc_id}_{원본명}에 저장
   │
   ├─▶ MinerU /file_parse 호출                mineru_client.py
   │      └─ content_list 블록 + 이미지(base64 data URI)
   │
   ├─▶ 이미지 블록만 PaddleOCR 보강            mineru_client.build_pages
   │      └─ '[이미지 텍스트]' 접두어로 본문에 인라인 삽입
   ├─▶ 머리글·페이지번호 등 페이지 부속물 제거   (§4.5)
   ├─▶ 페이지 단위 Document 조립               document_processor.load_pdf
   │
   ├─▶ 원본 파싱 결과를 JSON으로 별도 저장       parsed_store.save
   │      └─ 문서 뷰어 탭 전용 (읽기 전용 사이드 채널)
   │
   ├─▶ 페이지 병합 → 청킹                     chunking.merge_pages → build_splitter
   └─▶ KURE-v1 임베딩 → ChromaDB 적재          vector_store.add_documents
```

MinerU 호출이 실패하면 예외를 삼키고 `PyPDFLoader`로 폴백한다. 파싱 서비스가 죽어
있다는 이유만으로 업로드 자체가 실패하지 않게 하려는 의도다.

### 3.2 질의 응답 (Retrieval)

```
질문 + conversation_id + 선택된 document_ids
   │
   ├─▶ RAGService.ask_question (LCEL)
   │     ├─ 1단계: 대화 이력이 있을 때만 독립형 질문으로 재작성 (LLM 호출 #1)
   │     ├─ 2단계: ScoringRetriever로 유사도 검색 (k=10, document_id 필터)
   │     └─ 3단계: page_content를 빈 줄로 이어 QA_PROMPT에 주입 (LLM 호출 #2)
   │
   └─▶ answer + sources[] (문서명·페이지·유사도 점수·이미지 URL)
```

한 번의 질문에 **LLM이 최대 두 번 호출된다**. 첫 호출은 "그가 하는 일은?" 같은
대명사 의존 후속 질문을 검색 가능한 독립 질문으로 바꾸기 위한 것이며, **대화의 첫
질문에서는 이력이 비어 있으므로 이 단계를 건너뛴다.**

검색과 답변 모두 재작성된 질문을 쓴다. 컨텍스트는 청크마다 `[출처: 문서명, p.N]`
헤더를 한 줄 붙여 빈 줄로 이어붙인다. 헤더 형식은 `QA_PROMPT`가 요구하는 인용
형식과 같게 두어, LLM이 지어내지 않고 그대로 옮겨 적을 수 있게 했다.

---

## 4. 주요 설계 결정

### 4.1 한국어 임베딩으로 KURE-v1 선택

`nlpai-lab/KURE-v1`을 `normalize_embeddings=True`로 GPU(`cuda`)에서 구동한다.
컬렉션 이름도 `documents_kure_v1`로 모델명을 박아두었다 — 임베딩 모델을 바꾸면
기존 벡터와 차원·의미 공간이 달라져 섞이면 안 되기 때문에, 컬렉션 자체를 분리한다.

### 4.2 PaddleOCR을 별도 프로세스로 격리

**PaddleOCR은 torch가 이미 로드된 프로세스에서 segfault를 낸다.** paddlex가
modelscope를 import하고, modelscope가 torch를 import하는 경로 때문이다. 그런데 이
API 프로세스는 임베딩 모델 때문에 이미 torch를 들고 있다.

그래서 `SubprocessOCRService`가 `ProcessPoolExecutor(max_workers=1)`로 워커를 하나
띄우고 재사용한다. 컨텍스트는 반드시 **`spawn`** 이다 — `fork`면 자식이 부모의
torch를 그대로 상속받아 격리 목적이 무너진다. 워커가 죽으면(`BrokenProcessPool`)
새로 띄워 한 번 재시도하고, 타임아웃(기본 120초)이면 폐기한다.

### 4.3 ScoringRetriever — 유사도 점수 보존

LangChain의 `vector_store.as_retriever()`는 문서만 돌려주고 **점수를 버린다.**
출처 UI에 유사도를 표시하려면 별도로 재검색해야 하는데, 그러면 답변 생성에 쓰인
검색 결과와 다를 수 있다. `ScoringRetriever`는
`similarity_search_with_relevance_scores`를 직접 호출해 점수를
`doc.metadata['similarity_score']`에 실어 보내므로, **답변과 출처가 동일한 검색
결과에서 나온다는 것이 보장된다.**

### 4.4 시멘틱 청킹 직접 포팅

`chunking.py`는 Greg Kamradt의 *5 Levels of Text Splitting* (Level 4) 알고리즘을
직접 구현했다. `langchain_experimental.SemanticChunker`를 쓰지 않은 이유는 알고리즘을
"experimental" 패키지 뒤에 두지 않고 직접 소유하기 위해서다. 앱의 KURE-v1 임베딩을
재사용하므로 모델이 두 번 로드되지 않는다.

동작: 문장 분리 → 앞뒤 이웃과 묶어 임베딩(노이즈 완화) → 인접 임베딩 간 코사인
거리 → 거리가 특정 백분위를 넘으면 경계로 간주. 코사인 거리는 scikit-learn 의존성을
추가하지 않으려고 numpy로 직접 계산한다.

전략은 업로드 시 `chunking_strategy` 폼 필드로 `default`(문자 수 기반) / `semantic`
중 선택한다.

### 4.5 페이지 부속물 제거와 페이지 병합

실측에서 드러난 두 가지 구조적 문제를 다룬다. 조사 과정과 측정치는
[검색 결과가 머리글 조각으로만 채워지는 문제](docs/troubleshooting/2026-09-01-rag-retrieval-quality.md)에
정리되어 있다.

**(a) 머리글·페이지 번호가 본문을 오염시킨다.** 545페이지 PDF에서 책 제목이 거의
모든 페이지에 반복되어 전체 청크의 57%가 같은 문자열로 시작했다. 임베딩이 그
보일러플레이트에 지배되면서, 어떤 질문을 하든 본문이 아니라 **머리글에 매칭**되어
거의 동일한 조각들이 검색됐다.

대응은 두 겹이다. MinerU가 붙여주는 `header`/`footer`/`page_number` 타입을 걸러내고,
**동작 기반 탐지**를 덧붙인다 — 80자 이하이면서 전체 페이지의 20% 이상(최소 5페이지)에
똑같이 등장하는 문자열은 러닝 헤더로 본다. 타입 필터만으로 부족한 이유는 실측에서
러닝 헤더 546개 중 430개를 MinerU가 평범한 `text`로 분류했기 때문이다. 두 조건을
모두 요구하므로 우연히 반복되는 짧은 구절이나 반복되는 긴 문단은 살아남는다.

**(b) 청크가 한 페이지 안에 갇힌다.** 두 splitter 모두 Document를 하나씩 처리하고
`load_pdf`는 페이지당 Document를 하나씩 만든다. 큰 활자 레이아웃이라 페이지당 200자
남짓이던 문서에서는 `chunk_size=1000`에 **영영 도달할 수 없었다**(중앙값 158자).

`merge_pages`가 분할 **전에** 같은 파일의 연속 페이지를 목표 크기까지 합친다. 병합
시 첫 페이지의 메타데이터를 유지하되 `image_ids`는 합집합, OCR 플래그는 OR/합계로
이어붙인다 — 그러지 않으면 병합된 청크가 인용하는 도표를 잃는다.

또한 시멘틱 청킹의 백분위 임계값은 **문서 전체 거리 분포에서 한 번** 계산한다.
백분위는 상대 지표라서 문서마다 따로 구하면 내용과 무관하게 비슷한 개수의 경계가
강제로 생기고, 문장 두세 개짜리 페이지에서는 주제 전환이 없는데도 잘린다.
`min_chunk_chars`(기본 200)로 너무 짧은 청크를 만드는 경계는 다음 청크로 넘긴다.

### 4.6 Chroma 메타데이터의 스칼라 제약

Chroma는 메타데이터 값으로 `str`/`int`/`float`/`bool`만 받는다. `None`도 거부한다.
따라서 이미지 ID 목록은 콤마로 join한 문자열로 저장한다(ID가 md5 hexdigest라 콤마가
들어갈 일이 없어 무손실). 마찬가지로 `chunk_size`/`chunk_overlap`은 `default` 전략에만
해당하므로, `semantic`일 때는 `None`을 쓰는 대신 **키 자체를 넣지 않는다.**

### 4.7 저장소가 곧 레지스트리

문서 메타데이터용 DB가 없다. 대신 각 저장 디렉터리가 사실상의 레지스트리다.

- 문서 수 상한(기본 10개) 판정은 `upload_dir`을 센다. 파일명이
  `{document_id}_{원본명}` 규칙이고, 파일 타입이나 파싱 성공 여부와 무관하게
  성공한 업로드마다 항상 항목이 하나 있는 유일한 디렉터리이기 때문이다.
- 문서 뷰어의 목록은 `parsed_storage_dir`의 JSON 파일들을 훑어 만든다.

이 때문에 `GET /api/v1/documents/`는 **빈 배열을 반환하는 스텁**이고, 프론트엔드는
실제로 `GET /api/v1/documents/parsed`를 문서 목록으로 쓴다.

삭제(`DELETE /{document_id}`)는 네 곳을 모두 지운다: Chroma 청크, 저장된 이미지,
파싱 결과 JSON, 업로드 원본 파일. Chroma 삭제는 `vector_store.delete(where=...)`를
그대로 쓴다. langchain-community 0.0.38에서는 이 호출이 `where=`를 조용히 버려
`_collection`을 직접 호출해야 했으나, 0.4.2는 `**kwargs`를 그대로 넘긴다.

### 4.8 대화 메모리는 인메모리

`RAGService.conversation_histories`가 `conversation_id → list[BaseMessage]`
딕셔너리를 들고 있다. **프로세스가 재시작되면 전부 사라지고, 워커를 여러 개 띄우면
공유되지 않는다.**

---

## 5. 저장소 레이아웃

```
backend/app/storage/
├── uploads/      {document_id}_{원본파일명}      — 원본 (문서 수 상한의 기준)
├── chroma_db/    chroma.sqlite3 + 인덱스          — 청크·임베딩·메타데이터
├── parsed/       {document_id}.json              — MinerU 원본 블록 (뷰어용)
└── images/       {document_id}/{image_id}.png    — 추출 이미지 (출처 인용용)
```

`image_id`는 이미지 바이트의 md5 앞 16자다. 같은 이미지가 여러 페이지에 나오면
OCR을 한 번만 수행하고 결과를 캐시한다.

---

## 6. API


| 메서드 | 경로                                       | 설명                                    |
| -------- | -------------------------------------------- | ----------------------------------------- |
| POST   | `/api/v1/upload/`                          | 단일 파일 업로드·처리                  |
| POST   | `/api/v1/upload/batch`                     | 다중 업로드 (실패해도 나머지 계속 진행) |
| POST   | `/api/v1/chat/`                            | 질의 → 답변 + 출처                     |
| DELETE | `/api/v1/chat/conversation/{id}`           | 대화 이력 삭제                          |
| GET    | `/api/v1/documents/parsed`                 | 파싱된 문서 목록 **(실질적 문서 목록)** |
| GET    | `/api/v1/documents/{id}/parsed`            | 페이지별 파싱 블록 상세                 |
| GET    | `/api/v1/documents/{id}/images/{image_id}` | 추출 이미지                             |
| GET    | `/api/v1/documents/`                       | 스텁 — 항상 `[]`                       |
| DELETE | `/api/v1/documents/{id}`                   | 문서와 전 산출물 삭제                   |

Swagger UI: `http://localhost:8000/docs`

---

## 7. 설정 (`backend/.env`)

기본값은 `app/config.py`의 `Settings`에 있다. 자주 건드리는 값만 추린다.


| 키                                       | 기본값                       | 의미                         |
| ------------------------------------------ | ------------------------------ | ------------------------------ |
| `llm_model`                              | `gemma4:26b-a4b-it-q4_K_M`   | Ollama 모델명                |
| `ollama_base_url`                        | `http://192.168.0.169:11434` | **원격 호스트 기본값**       |
| `embedding_model`                        | `nlpai-lab/KURE-v1`          | 임베딩 모델                  |
| `embedding_device`                       | `cuda`                       | `cpu`로 바꾸면 크게 느려짐   |
| `retrieval_k`                            | `10`                         | 검색할 청크 수               |
| `chunk_size` / `chunk_overlap`           | `1000` / `200`               | 페이지 병합 목표 크기도 겸함 |
| `chunking_strategy`                      | `default`                    | `default` \| `semantic`      |
| `semantic_chunker_breakpoint_percentile` | `95.0`                       | 경계 임계 백분위             |
| `semantic_chunker_min_chunk_chars`       | `200`                        | 최소 청크 길이               |
| `max_documents`                          | `10`                         | 동시 보유 문서 수 상한       |
| `max_upload_size`                        | `10MB`                       |                              |
| `mineru_base_url`                        | `http://127.0.0.1:8100`      |                              |
| `mineru_lang_list`                       | `["korean"]`                 |                              |
| `ocr_device`                             | `cpu`                        | PaddleOCR 실행 디바이스      |
| `ocr_isolate_process`                    | `true`                       | §4.2 — 끄면 segfault 위험  |

`ollama_base_url` 기본값이 사설 IP이므로 단독 실행 시 반드시 `.env`에서 덮어써야 한다.

---

## 8. 프론트엔드 구조

탭 두 개짜리 단일 페이지다.

- **채팅 탭** — `FileUploader`(드래그 앤 드롭), `DocumentSelector`(검색 대상 문서
  선택), `ChatWindow` → `Message` → `SourceCitation`(펼침형 출처, 유사도·페이지·
  도표 이미지 표시)
- **문서 뷰어 탭** — `ParsedDocumentViewer`. MinerU 파싱 결과를 페이지·블록 단위로
  렌더링한다. 표 블록은 `table_body`(HTML 추출)와 `image_id`(MinerU가 실제로 파싱한
  스크린샷)를 **둘 다** 보여줘 사용자가 대조할 수 있게 한다.

탭 전환은 언마운트가 아니라 `hidden` 속성으로 처리한다. 따라서 마운트 시 1회
fetch하는 컴포넌트는 탭을 오가도 갱신되지 않는다 — 문서 목록을 `App.tsx`가 단독
소유하고 props로 내려주는 이유다.

MinerU의 `table_body`는 HTML 문자열이라 `dangerouslySetInnerHTML`로 렌더링해야
하는데, 그 전에 **DOMPurify로 반드시 sanitize한다.**

---

## 9. 알려진 제약

1. **인증 없음.** 누구나 모든 문서를 읽고 지울 수 있다.
2. **대화 이력 비영속.** 재시작하면 사라지고, 다중 워커 환경에서 공유되지 않는다.
3. **문서 메타데이터 DB 없음.** 파일시스템이 레지스트리 역할을 대신한다(§4.7).
4. **단일 인스턴스 전제.** OCR 워커·인메모리 메모리·`lru_cache` 기반 싱글턴이 모두
   프로세스 로컬이다.
5. **문서 10개 상한.** 로컬 개발 환경 보호용 값이다.
6. **문장 분리기가 마침표 계열에 의존한다.** `(?<=[.?!])\s+` 정규식이라, 마침표 없이
   줄바꿈으로만 구성된 문서에서는 시멘틱 청킹이 사실상 동작하지 않는다.
7. **MinerU·Ollama가 외부 프로세스다.** 둘 다 별도로 띄워야 하며, MinerU는 죽으면
   폴백되지만 Ollama가 죽으면 채팅이 실패한다.

---

## 10. 개발

```bash
# 백엔드
cd backend
pytest                                   # 149 tests
uvicorn app.main:app --reload --port 8000

# 프론트엔드
cd frontend
npm run build                            # tsc 타입 체크 포함
npm run dev
```

`npm run lint` 스크립트는 정의되어 있지만 ESLint 설정 파일이 저장소에 없어 현재는
동작하지 않는다.

설계 문서와 구현 계획은 `docs/superpowers/{specs,plans}/`에 있다.
