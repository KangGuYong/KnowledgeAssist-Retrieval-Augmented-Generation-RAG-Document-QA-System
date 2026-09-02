# LangChain 1.x 마이그레이션

- 작성일: 2026-09-02
- 상태: 설계 검토 대기
- 관련 코드: `requirements.txt`, `rag_service.py`, `vector_store.py`, `document_processor.py`, `chunking.py`, `config.py`
- 선행 작업: [검색 결과 재배치](2026-09-02-search-result-reordering-design.md)

## 1. 문제

### 1.1 현재 핀이 왜 여기에 있는가

`backend/requirements.txt`는 `langchain==0.1.20` / `langchain-community==0.0.38`에
묶여 있다. 이 값의 유래는 두 커밋에 남아 있다.

- 최초 `0.1.0` 핀은 `f335f83 project structure`(2024-04-07)에서 스캐폴딩과 함께
  들어왔고, 선택 이유에 대한 기록은 없다. 당시엔 0.1이 현행 라인이었다.
- `13b9419`가 유일하게 문서화된 버전 결정이다. `SemanticChunker`를 쓰려고
  `langchain-experimental==0.0.55`를 넣으면서, 그 패키지의 상위 버전이 요구하는
  **0.2.x 파괴적 점프를 피하려고** 0.1.x 라인에 머물기로 했다.

**그 이유는 이미 만료됐다.** `4e9269c`가 시멘틱 청킹을 직접 포팅하며
`langchain-experimental`을 의존성에서 완전히 제거했고, 커밋 메시지도 "removes […]
its version constraints entirely"라고 명시한다. 0.1.x를 붙잡던 유일한 제약이 그때
끊겼지만 핀은 그대로 남았다.

`ARCHITECTURE.md`의 `0.1.x 라인 유지`는 그래서 **이유가 아니라 현상 기술**이다.

### 1.2 도착지가 없는 라인에 서 있다

| 라인 | 마지막 릴리스 | 상태 |
|---|---|---|
| `langchain` 0.1.x | **0.1.20 (2024-05-10)** | 종료 — 우리가 쓰는 버전이 그 라인의 마지막 |
| `langchain` 0.2.x | 0.2.17 (2024-11-04) | 종료 |
| `langchain` 0.3.x | 0.3.30 (2026-05-07) | 사실상 종료 |
| `langchain` 1.3.x | **1.3.18 (2026-08-27)** | 현행 |
| `langchain-community` 0.0.x | **0.0.38 (2024-05-08)** | 종료 |
| `langchain-community` 0.4.x | **0.4.2 (2026-05-22)** | 현행 (community는 1.0에 합류하지 않음) |

LangChain은 별도의 LTS 갈래를 운영하지 않는다. `1.0`(2025-10-17) 이후 1.1 → 1.2 →
1.3으로 이어지는 단일 현행 라인이 곧 안정판이다.

### 1.3 취약점

OSV API 조회 결과(2026-09-02 기준):

| 패키지 | 취약점 |
|---|---|
| `langchain==0.1.20` | **9건** |
| `langchain-community==0.0.38` | **8건** (HIGH 2 포함) |

`langchain-community`의 HIGH 2건은 FAISS pickle 역직렬화를 통한 임의 명령 실행
(수정 `0.2.4`)과 EverNoteLoader XXE를 통한 로컬 파일 유출(수정 `0.3.27`)이다.
나머지는 WebResearchRetriever SSRF(수정 `0.2.9`), DoS(수정 `0.2.5`) 등이다.

**실제 노출도는 낮다.** 취약한 코드 경로가 FAISS·EverNoteLoader·WebResearchRetriever
인데, 이 프로젝트가 community에서 쓰는 표면은 `ChatOllama`, `HuggingFaceEmbeddings`,
`Chroma`, PDF/TXT/DOCX `document_loaders`, `LongContextReorder`뿐이라 해당 경로를
타지 않는다.

즉 **긴급 패치 사안이 아니라, 취약점 0건 라인으로 갈 수 있는데 8~9건짜리에 머물러
있는 상태**다. 이 작업의 동기는 사고 대응이 아니라 부채 정리다.

## 2. 설계 원칙

- **경유하지 않는다.** 어차피 치러야 할 비용이 같다면, 이미 멈춘 라인을 중간
  기착지로 삼지 않는다(근거는 §3.1).
- **의존성 범프와 코드 재작성을 같은 커밋에 넣지 않는다.** `13b9419`가 이미 쓴
  방식이다. 회귀가 났을 때 원인이 의존성인지 재작성인지 헷갈리지 않아야 한다.
- **동작 동등성을 먼저 확보한다.** 1단계에서는 기능을 바꾸지 않는다. 기존 테스트가
  수정 없이 그대로 통과하는 것이 성공 기준이다.
- **레거시 shim은 종착지가 아니라 발판이다.** `langchain-classic`은 단계를 쪼개기
  위해 쓰고, 계획된 후속 단계에서 걷어낸다.
- **검증 가능한 사실만 근거로 쓴다.** 이 문서의 버전·모듈 존재 여부는 전부 PyPI
  휠을 직접 열고 OSV API를 조회해 확인했다. 재확인 방법은 §7.4에 남긴다.

## 3. 설계

### 3.1 두 경로 비교와 채택 근거

두 경로를 검토했다.

**경로 A — 0.3.x 경유**: `langchain` 0.3.30 / `core` 0.3.86 / `community` 0.3.31 /
`text-splitters` 0.3.11.

**경로 B — 1.3.x 직행**: `langchain` 1.3.18 / `core` 1.6.1 / `community` 0.4.2 /
`text-splitters` 1.1.2 + `langgraph` 1.2.11 + `langchain-ollama` 1.1.0.

`langchain` 0.3.30 휠을 열어 확인한 결과, 레거시 shim이 전부 살아 있다 —
`text_splitter.py`(`"""Kept for backwards compatibility."""`로 시작하는 재export),
`schema/retriever.py`, `callbacks/manager.py`, `memory/buffer.py`,
`chains/conversational_retrieval/`. 즉 **경로 A는 앱 코드 import 변경이 0건**이다.

그럼에도 경로 A를 채택하지 않는다. 이유는 셋이다.

**(1) 가장 위험한 비용을 양쪽이 똑같이 치른다.** `langchain` 0.3.30은
`pydantic>=2.7.4`를, `langchain-community` 0.3.31은 `pydantic-settings>=2.10.1`을
요구한다. 현재 핀은 `pydantic==2.5.0` / `pydantic-settings==2.1.0`이다. 1.3.18도
같은 `pydantic>=2.7.4`를 요구하므로, **이 강제 범프는 경로 선택과 무관하게
발생한다.** 마이그레이션에서 가장 넓게 파급되는 부분(FastAPI 0.109 호환,
`config.py`의 구식 `class Config:`, pydantic 모델인 `ScoringRetriever`)이 여기다.

**(2) 경로 A는 취약점 0건에 도달하지 못한다.**

| 패키지 | 취약점 |
|---|---|
| `langchain==0.3.30` | 2건 (`GHSA-gr75-jv2w-4656` 등) |
| `langchain-core==0.3.86` | 4건 (`GHSA-qh6h-p6c9-ff54` HIGH 포함) |
| `langchain-community==0.3.31` | 0건 |
| 경로 B의 모든 대상 (`langchain` 1.3.18 / `core` 1.6.1 / `community` 0.4.2 / `classic` 1.0.8 / `langchain-ollama` 1.1.0) | **0건** |

잔존 3건의 수정 버전은 각각 `langchain 1.3.9`, `langchain-core 1.2.11`,
`langchain-core 1.2.22`로 **전부 1.x에만 있다.** 0.3 라인은 `core 0.3.86`에서
멈춘 채 패치되지 않는다.

이 3건 역시 실제 노출은 없다(`load_prompt` 경로 순회는 인라인
`PromptTemplate(template=...)`만 쓰는 우리와 무관, SSRF는 `ChatOpenAI` 전용, 파일
검색 미들웨어는 미사용). 그러나 **감사 지표상 0건에 도달할 수 있는데 6건을 남기는
선택**이 된다.

**(3) 결국 다시 해야 한다.** 경로 A의 도착지는 이미 멈춘 라인이므로, 언젠가 경로 B의
작업을 그대로 반복하게 된다.

### 3.2 단계 분할

세 커밋으로 나눈다.

| 단계 | 내용 | 성공 기준 |
|---|---|---|
| **0** | `pydantic` / `pydantic-settings` 범프만 | 코드 변경 없이 전체 테스트 통과 |
| **1** | LangChain 1.3.18 + `langchain-classic`로 **import만 이동** | 체인 로직 무변경, 기존 테스트 그대로 통과 |
| **2** | `ConversationalRetrievalChain` → LCEL 재작성, `langchain-classic` 제거 | 별도 설계 문서로 분리 |

이 문서는 **0단계와 1단계**를 다룬다. 2단계는 동작이 바뀔 수 있으므로 범위에서
제외한다(§4).

### 3.3 1단계의 지름길 — `langchain-classic`

`langchain` 1.3.18 휠에는 `chains`, `memory`, `prompts`, `schema`, `callbacks`가
**하나도 없다.** 전부 삭제됐다.

그런데 `langchain-community` 0.4.2가 `langchain-classic>=1.0.7`을 **이미 의존성으로
끌고 온다.** 그리고 `langchain-classic` 1.0.8(2026-06-10)에는
`chains/conversational_retrieval/`, `memory/buffer.py`, `prompts`, `retrievers`가
그대로 들어 있다.

따라서 1단계는 **LCEL 전면 재작성 없이** `from langchain.chains import …` →
`from langchain_classic.chains import …` 수준의 이동으로 끝난다. 새 패키지를
추가로 설치할 필요도 없다(community가 끌고 온다).

이것이 부채를 새 패키지로 옮겨 담는 것임은 분명하다. 그래서 2단계를 계획에
명시해 두고, `langchain-classic`을 **한시적 발판**으로만 쓴다.

### 3.4 파일별 수정 지점 (1단계)

| 파일 | 현재 | 1단계 이후 |
|---|---|---|
| `services/chunking.py` | `langchain.schema.Document`<br>`langchain.text_splitter.RecursiveCharacterTextSplitter`<br>`langchain_core.embeddings.Embeddings` | `langchain_core.documents.Document`<br>`langchain_text_splitters.…`<br>변경 없음 |
| `services/document_processor.py` | `langchain.schema.Document`<br>`langchain_community.document_loaders.…` | `langchain_core.documents.Document`<br>변경 없음 |
| `services/vector_store.py` | `langchain.schema.Document`<br>`community.vectorstores.Chroma`<br>`community.embeddings.HuggingFaceEmbeddings` | `langchain_core.documents.Document`<br>변경 없음<br>변경 없음 |
| `services/rag_service.py` | `langchain.chains.ConversationalRetrievalChain`<br>`langchain.memory.ConversationBufferMemory`<br>`langchain.prompts.PromptTemplate`<br>`langchain.schema.Document`<br>`langchain.schema.retriever.BaseRetriever`<br>`langchain.callbacks.manager.CallbackManagerForRetrieverRun`<br>`community.chat_models.ChatOllama`<br>`community.document_transformers.LongContextReorder` | `langchain_classic.chains.…`<br>`langchain_classic.memory.…`<br>`langchain_core.prompts.…`<br>`langchain_core.documents.…`<br>`langchain_core.retrievers.BaseRetriever`<br>`langchain_core.callbacks.…`<br>**`langchain_ollama.ChatOllama`**<br>변경 없음 |
| 테스트 5개 | `langchain.schema.Document` 등 | `langchain_core.documents.Document`로 일괄 치환 |

위 표의 대상 심볼은 전부 휠에서 export를 확인했다 — `langchain_core` 1.6.1의
`documents.Document` / `prompts.PromptTemplate` / `retrievers.BaseRetriever` /
`callbacks.CallbackManagerForRetrieverRun`, `langchain_classic` 1.0.8의
`chains.ConversationalRetrievalChain` / `memory.ConversationBufferMemory`,
`langchain_ollama` 1.1.0의 `ChatOllama`.

**부담은 `rag_service.py` 하나에 몰려 있다.** 나머지 파일은
`langchain.schema.Document` → `langchain_core.documents.Document` 기계적 치환뿐이다.

### 3.5 `ChatOllama`는 partner 패키지로 이동한다

`langchain-community` 0.4.2 휠에 `chat_models/ollama`가 **없다.** `ChatOllama`는
`langchain-ollama` 패키지(1.1.0, 2026-04-07)로 이관됐다. 1단계에서 이 의존성을
새로 추가한다.

같은 휠에서 `embeddings/huggingface`, `vectorstores/chroma`,
`document_transformers/long_context_reorder`, `document_loaders/pdf`는 **모두 존재를
확인했다.** 이들은 deprecation 경고가 붙을 수 있으나 1단계에서는 그대로 둔다.

### 3.6 `LongContextReorder`는 그대로 살아 있다

[검색 결과 재배치 설계 문서 §6](2026-09-02-search-result-reordering-design.md)은
"상위 버전에서 이 클래스의 import 경로가 이동했으므로 의존성을 올릴 때 함께
손봐야 한다"고 기록했다.

**재확인 결과 이 우려는 해소됐다.** 문제의 이동은
`langchain.document_transformers` → `langchain_community.document_transformers`
였고, 이 프로젝트는 **이미 community 쪽 경로를 쓰고 있다.** `community` 0.4.2 휠에
`document_transformers/long_context_reorder.py`가 존재하며 `LongContextReorder`가
그대로 export된다.

남는 일은 계약 테스트(`test_long_context_reorder.py`)를 새 버전에서 돌려 배치
순서가 동일한지 확인하는 것뿐이다. 그 테스트를 둔 목적이 정확히 이것이다.

### 3.7 함께 해소되는 부채 — Chroma `where=` 우회

`langchain-community`의 `Chroma.delete()` 소스를 버전별로 대조했다.

```python
# 0.0.38 (현재) — kwargs를 버린다
self._collection.delete(ids=ids)

# 0.4.2 — where= 가 전달된다
self._collection.delete(ids=ids, **kwargs)
```

따라서 `vector_store.py`의 `self.vector_store._collection.delete(where=...)`
사설 API 우회와 그 이유를 설명하는 주석·`ARCHITECTURE.md` 문단은 **1단계 이후
불필요해진다.**

다만 **1단계에서는 걷어내지 않는다.** 1단계의 성공 기준은 동작 동등성이고, 이
우회는 상위 버전에서도 정상 동작한다(`_collection.delete(where=...)`는 여전히
유효). 별도 커밋으로 분리해 되돌리기 쉽게 둔다. 관련 회귀 테스트
(`test_vector_store.py`)도 그때 함께 정리한다.

## 4. 범위

**포함**

- `pydantic` / `pydantic-settings` 범프 (0단계)
- `requirements.txt`의 LangChain 계열 핀 갱신 (1단계)
- 앱 코드·테스트의 import 경로 이동 (1단계, §3.4)
- `langchain-ollama` 의존성 추가 (1단계)
- `ARCHITECTURE.md` / `README.md`의 버전 표와 `0.1.x 라인 유지` 문구 갱신

**제외**

- **LCEL 재작성 (2단계).** `ConversationalRetrievalChain`·`ConversationBufferMemory`
  제거와 `langchain-classic` 탈피는 동작이 바뀔 수 있어 별도 설계 문서로 다룬다.
- **Agent Middleware 도입.** [wikidocs 4-1](https://wikidocs.net/318926)의 미들웨어는
  `create_agent` 기반 에이전트 루프에서만 발화한다. 이 프로젝트는 루프 없는 고정
  2단계 파이프라인이라 훅을 걸 자리가 없다. 2단계 이후 재검토 대상이다.
- **`chromadb` 범프.** 별개 판단이 필요하다 — CRITICAL 1건 + HIGH 1건이 `0.4.17`
  이후 **최신 `1.5.9`까지 미수정**(`last_affected: 1.5.9`)이다. 둘 다 서버 모드의
  인증·테넌트 권한 문제라 임베디드 영속 클라이언트로 쓰는 이 프로젝트에는
  해당하지 않으며, **올려도 사라지지 않는다.** 버전 선택 기준으로 삼을 이유가 없다.
- **`Chroma`/`HuggingFaceEmbeddings`의 partner 패키지 이전.** `langchain-chroma`,
  `langchain-huggingface`로 옮기는 일은 community 0.4.2에서 아직 동작하므로
  미루고, deprecation 경고를 관찰한 뒤 판단한다.
- **기능 변경.** 조건부 OCR 프롬프트, LLM 호출 재시도 등은 2단계에서 다룬다.

## 5. 설정

`backend/requirements.txt` 변경 예정 내역. 단계별로 나눠 커밋한다.

```
# 0단계
pydantic==2.5.0          → >=2.7.4
pydantic-settings==2.1.0 → >=2.10.1

# 1단계
langchain==0.1.20          → 1.3.18
langchain-community==0.0.38 → 0.4.2
langchain-ollama            → 1.1.0   (신규)
# 전이 의존성으로 함께 들어옴: langchain-core 1.6.1, langgraph 1.2.11,
#                            langchain-text-splitters 1.1.2, langchain-classic 1.0.8
```

애플리케이션 설정(`config.py`)에 **추가·제거되는 항목은 없다.**

## 6. 한계

- **`pydantic` 범프가 이 작업에서 가장 넓은 파급면이다.** LangChain 자체보다
  `config.py`의 `class Config:`(pydantic-settings 2.x에서 `SettingsConfigDict`로
  대체 권장), pydantic 모델인 `ScoringRetriever`, FastAPI 0.109와의 조합이 더
  위험하다. 0단계를 독립 커밋으로 분리하는 이유다.
- **`langchain-classic`은 부채를 옮겨 담는 것이다.** 1단계 종료 시점의 코드는
  1.x 위에서 돌지만 여전히 레거시 체인 API를 쓴다. 2단계 없이는 절반만 끝난다.
- **deprecation 경고가 대량으로 발생할 수 있다.** `HuggingFaceEmbeddings`,
  `Chroma`, community `document_loaders` 등이 partner 패키지 이전을 권고한다.
  1단계에서는 경고를 남겨 두고 목록만 기록한다.
- **1.x의 실제 런타임 동작 차이는 검증되지 않았다.** 휠의 모듈 존재 여부는
  확인했지만, `langchain-classic`의 `ConversationalRetrievalChain`이 1.6.1 코어
  위에서 0.1.53 코어와 동일하게 동작하는지는 테스트를 돌려 봐야 안다.
- **취약점 8~9건 중 실제로 우리를 위협하던 것은 없다.** 이 작업의 보안 효과는
  실질적 위험 제거가 아니라 감사 지표 개선이다. 과장하지 않는다.
- **`chromadb`의 미수정 CRITICAL 1건은 이 작업으로 사라지지 않는다.** 범위에서
  제외했을 뿐 존재하지 않는 것이 아니다(§4).

## 7. 검증

### 7.1 단계별 게이트

각 단계는 **다음 단계로 넘어가기 전에 전체 테스트를 통과해야 한다.**

| 단계 | 통과 기준 |
|---|---|
| 0 | 코드 변경 0줄, 전체 테스트 통과. 실패 시 원인은 100% pydantic이다. |
| 1 | import 경로 외 로직 변경 0줄, 전체 테스트 통과. |

### 7.2 계약 테스트가 1차 방어선이다

`test_long_context_reorder.py`는 `LongContextReorder`의 홀수/짝수 배치 순서를
`2 4 6 8 10 9 7 5 3 1` 형태로 못박아 둔 라이브러리 계약 테스트다. 서드파티의
정확한 순서 동작에 의존하기로 한 결정을 지키는 장치이며, **이번 범프가 그
테스트의 첫 실전이다.** 여기서 실패하면 §3.6의 판단이 틀린 것이므로 진행을
멈추고 재설계한다.

`test_vector_store.py`의 `where=` 회귀 테스트는 §3.7의 우회가 상위 버전에서도
동작함을 확인한다.

### 7.3 수동 스모크

테스트만으로는 잡히지 않는 부분을 실제로 돌려 확인한다.

- 문서 업로드 → MinerU 파싱 → 인덱싱 (원격 Ollama·MinerU 필요)
- 질문 → 답변 + 출처 목록 (점수 내림차순 유지 확인)
- 후속 질문 → 대화 메모리가 standalone question 재작성에 반영되는지
  (`langchain-classic`의 메모리가 1.x 코어에서 동작하는지 확인하는 지점)
- 문서 삭제 → Chroma 청크·이미지·파싱 JSON·업로드 원본 네 곳 모두 제거

### 7.4 이 문서의 사실 재확인 방법

버전과 취약점 정보는 시간이 지나면 낡는다. 재확인 명령을 남긴다.

```bash
python3 -c "import json,urllib.request as u; d=json.load(u.urlopen('https://pypi.org/pypi/langchain/json')); print(d['info']['version'])"
```

```bash
python3 -c "import json,urllib.request as u; r=u.Request('https://api.osv.dev/v1/query',data=json.dumps({'package':{'name':'langchain','ecosystem':'PyPI'},'version':'0.1.20'}).encode(),headers={'Content-Type':'application/json'}); print(len(json.load(u.urlopen(r)).get('vulns',[])))"
```

모듈 존재 여부는 PyPI 휠을 내려받아 `zipfile.ZipFile(...).namelist()`로 확인했다.
`pip install` 없이 검증할 수 있으므로 환경을 더럽히지 않는다.

## 8. 참고

- [wikidocs 4-1. 미들웨어(Middleware) 개요](https://wikidocs.net/318926) — 1.x
  에이전트 미들웨어. 이번 범위에서는 제외(§4)했으나 2단계 이후 재검토 대상.
- 커밋 `13b9419` — 0.1.x 라인 유지 결정의 원문
- 커밋 `4e9269c` — 그 결정의 전제(`langchain-experimental`)를 제거한 커밋
