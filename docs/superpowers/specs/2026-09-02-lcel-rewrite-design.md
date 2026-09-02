# LCEL 재작성 (langchain-classic 탈피)

- 작성일: 2026-09-02
- 상태: 2A 구현 완료 (`bfad7e2`, `6ce2b79`). 2B는 네 항목 완료
  (`c34e9ac`, `3dee526`, `2d49620`), 한국어 condense 프롬프트만 남음
- 관련 코드: `rag_service.py`, `chat.py`, `config.py`, `requirements.txt`
- 선행 작업: [LangChain 1.x 마이그레이션](2026-09-02-langchain-1x-migration-design.md) (0·1단계 완료)

## 1. 문제

### 1.1 발판을 걷어내야 한다

[1단계](2026-09-02-langchain-1x-migration-design.md)는 LangChain 1.3.18로 올라갔지만
`ConversationalRetrievalChain`과 `ConversationBufferMemory`를 그대로 두고
`langchain-classic`으로 import만 옮겼다. 그 문서가 명시했듯 이는 **한시적 발판**이며,
1단계 종료 시점의 코드는 1.x 위에서 돌지만 여전히 레거시 체인 API를 쓴다.

이 문서는 그 발판을 걷어내는 작업을 다룬다.

### 1.2 소유하지 않은 채로 의존하고 있는 것들

체인을 직접 소유하지 않아 생긴 문제를 조사하면서 세 가지를 발견했다. 셋 다
재작성이 필요한 이유이자, 재작성 없이는 고칠 수 없는 것들이다.

**발견 1 — 프롬프트가 지킬 수 없는 규칙을 요구한다.**

`QA_PROMPT`의 응답 규칙 2번은 이렇게 적혀 있다.

```
2. **출처 인용**: 답변에 사용한 정보의 출처를 [출처: 문서명] 형식으로 표시하세요.
```

그런데 컨텍스트를 조립하는 `StuffDocumentsChain`의 기본
`document_prompt`는 `PromptTemplate.from_template("{page_content}")`이고,
`document_separator`는 `"\n\n"`이다. 즉 LLM이 받는 것은 **청크 본문을 빈 줄로
이어붙인 것뿐**이며, `filename`·`page`·`document_id`는 메타데이터에만 있고
컨텍스트에 들어가지 않는다(`document_processor.load_pdf`는
`page_content=page.text`만 넣는다).

**LLM은 문서명을 알 방법이 없다.** 이 규칙을 따르려면 지어내는 수밖에 없다.
출처 표시는 실제로는 `_format_sources`가 만드는 API 응답이 담당하고 있고,
프롬프트의 이 줄은 근거 없는 지시로 남아 환각을 유도한다.

**발견 2 — `settings.max_tokens`는 아무 데도 연결돼 있지 않다.**

`config.py`의 `max_tokens: int = 2000`은 코드 어디에서도 읽히지 않는다.
`ChatOllama`에 전달되지도 않고(`_initialize_llm`은 `base_url`·`model`·
`temperature`만 넘긴다), `ConversationalRetrievalChain`의 `max_tokens_limit`으로
쓰이지도 않는다. 후자는 설정되지 않아 `_reduce_tokens_below_limit`이 no-op이다.

즉 **답변 길이 제한도, 컨텍스트 토큰 상한도 실제로는 없다.** 설정 파일만 보면
둘 다 있는 것처럼 보인다.

**발견 3 — `ask_question`은 `async`인데 내부는 전부 동기다.**

```python
async def ask_question(...):
    ...
    result = qa_chain({"question": question})   # 동기 호출
```

`chat.py`의 엔드포인트도 `async def`이므로 이 요청은 이벤트 루프 위에서 실행되고,
**원격 Ollama에 대한 LLM 호출 2회와 임베딩·벡터 검색이 끝날 때까지 루프 전체가
멈춘다.** 동시 요청은 사실상 직렬화된다. `async` 시그니처가 그 사실을 가리고 있다.

### 1.3 재작성으로 해소되는 이전 문서의 제약

[검색 결과 재배치 설계 §6](2026-09-02-search-result-reordering-design.md)은
`atransform_documents`가 `NotImplementedError`를 던지므로 "비동기 리트리버로
전환하면 이 지점에서 실패한다"고 기록했다.

**`langchain-community` 0.4.2에서는 구현돼 있다.** 소스를 확인했다:

```python
async def atransform_documents(self, documents, **kwargs):
    return _litm_reordering(list(documents))
```

비동기 전환을 막던 제약은 사라졌다.

## 2. 설계 원칙

- **재작성과 개선을 섞지 않는다.** 이 작업의 1차 목표는 `langchain-classic` 제거이며,
  그 커밋은 **동작 동등**이어야 한다. §1.2의 발견들은 재작성이 *가능하게 만드는*
  후속 작업이지 이번 커밋의 내용이 아니다(§4에서 2A/2B로 분리).
- **먼저 계약을 문서화한 뒤 재현한다.** 지금 코드는 `ConversationalRetrievalChain`이
  무엇을 하는지 모른 채 결과에만 의존한다. §3.1이 그 동작을 명세로 고정하고,
  §3.2가 그것을 재현한다. 명세 없이 다시 쓰면 무엇이 바뀌었는지 알 수 없다.
- **소유할 것만 소유한다.** 체인 조립·프롬프트·메모리는 우리가 가져온다.
  검색·재배치·임베딩은 그대로 둔다. `ScoringRetriever`와 `_format_sources`는
  **한 줄도 바꾸지 않는다.**
- **추가 의존성을 들이지 않는다.** LangGraph 체크포인터로 메모리를 옮기지 않는다
  (§3.3). 지금 필요한 것은 리스트 하나다.
- **`async`가 거짓말이 되지 않게 한다.** 재작성 후 `ask_question`은 실제로
  비동기여야 한다(§3.4).

## 3. 설계

### 3.1 현재 체인의 동작 명세

`langchain-classic` 1.0.8 소스를 읽어 확정한 내용이다. 재작성은 이것을 재현한다.

```
inputs = {"question": q, "chat_history": memory가 돌려준 메시지 리스트}
   │
   ├─ (1) chat_history를 문자열로 접는다  [_get_chat_history]
   │       각 턴 앞에 "\nHuman: " / "\nAssistant: " 를 붙여 이어붙임
   │       (선행 개행 포함, 빈 content는 건너뜀)
   │
   ├─ (2) 문자열이 비어 있지 않을 때만 질문 재작성  [CONDENSE_QUESTION_PROMPT]
   │       첫 질문이면 이 LLM 호출 자체가 일어나지 않는다
   │
   ├─ (3) 재작성된 질문으로 검색  [retriever.invoke(new_question)]
   │       max_tokens_limit이 없으므로 문서 절삭 없음
   │
   ├─ (4) 컨텍스트 조립
   │       "\n\n".join(doc.page_content for doc in docs)   ← 메타데이터 없음
   │
   ├─ (5) 답변 생성  [QA_PROMPT]
   │       rephrase_question=True이므로 {question}에는 **재작성된 질문**이 들어간다
   │
   └─ 출력 {"answer": ..., "source_documents": docs}  이후 memory가 턴을 저장
```

**(2)의 프롬프트는 LangChain 기본값이고 영어다.** 이 프로젝트가 작성한 적 없다.

```
Given the following conversation and a follow up question, rephrase the follow up
question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
```

`in its original language` 덕분에 한국어 질문은 한국어로 재작성되지만, 지시문
자체는 검토된 적 없는 영어 기본값이라는 사실을 기록해 둔다. 재작성 시 **이 문구를
그대로 옮긴다**(§4의 2A). 한국어 프롬프트로 바꾸는 것은 동작 변경이다.

### 3.2 LCEL 구조

`RAGService` 안에 조립한다. 새 모듈을 만들지 않는다 — 옮겨오는 로직이
`ask_question` 한 메서드 분량이고, 흩어놓으면 §3.1의 순서 의존성이 보이지 않게 된다.

```
ainvoke
  │
  ├─ standalone_question
  │     chat_history 비어 있음  → question 그대로 (LLM 호출 없음)
  │     아니면                  → CONDENSE_PROMPT | llm | StrOutputParser()
  │
  ├─ docs = await retriever.ainvoke(standalone_question)
  │     ScoringRetriever 그대로. 점수 부착·재배치 전부 기존 코드.
  │
  ├─ context = "\n\n".join(d.page_content for d in docs)
  │
  └─ answer = await (QA_PROMPT | llm | StrOutputParser()).ainvoke(
                  {"context": context, "question": standalone_question})

반환 {"answer": answer, "sources": _format_sources(docs), ...}
```

`source_documents`를 별도 출력 키로 흘려보낼 필요가 없어진다. 검색 결과를 지역
변수로 들고 있다가 `_format_sources`에 직접 넘기면 된다. `RunnableParallel`로
answer와 docs를 함께 뽑아내는 LCEL 관용구는 **쓰지 않는다** — 여기서는 파이썬
지역 변수가 더 짧고 읽기 쉽다.

### 3.3 메모리는 리스트로 대체한다

`ConversationBufferMemory`가 실제로 하는 일은 메시지를 리스트에 쌓는 것뿐이다.
`memory_key`·`output_key`·`return_messages`는 전부 체인과의 인터페이스를 맞추기
위한 설정이고, 체인을 우리가 소유하면 필요 없다.

```python
self.conversation_histories: Dict[str, list[BaseMessage]] = {}
```

턴 종료 시 `HumanMessage(question)`와 `AIMessage(answer)`를 append 한다.

**LangGraph 체크포인터를 쓰지 않는 이유.** `langgraph`는 이제 `langchain`의 하드
의존성이라 추가 설치 비용은 없다. 그러나 현재 메모리는 이미 인메모리·비영속이고
(`ARCHITECTURE.md` §4.8), `MemorySaver`가 주는 보장도 정확히 같다. 얻는 것 없이
개념과 코드가 늘어난다. **영속화가 필요해지면 그것은 저장소 선택 문제이지 LCEL
전환의 일부가 아니다.**

`clear_conversation`은 딕셔너리 키를 지우는 동작 그대로다.

### 3.4 비동기

LLM 호출 두 번을 `ainvoke`로 바꾸면 원격 Ollama 대기 중 이벤트 루프가 풀린다.
이것이 §1.2 발견 3에 대한 답이다.

검색은 완전한 비동기가 되지 않는다. `ScoringRetriever`는
`_get_relevant_documents`(동기)만 구현하므로 `ainvoke`는 `BaseRetriever`의 기본
스레드풀 폴백을 탄다. 그 아래 chromadb도 동기다. **`_aget_relevant_documents`를
추가하지 않는다** — 스레드풀로 내려가는 것은 동일하고, 두 벌의 검색 경로를
유지·테스트해야 하는 비용만 생긴다.

핵심은 "검색까지 비동기로 만드는 것"이 아니라 **"수 초짜리 LLM 대기가 루프를 막지
않게 하는 것"** 이다.

### 3.5 프롬프트를 소유한다

`CONDENSE_PROMPT`를 `rag_service.py`에 모듈 상수로 둔다. `QA_PROMPT` 바로 옆이며,
문구는 §3.1의 영어 기본값을 그대로 옮긴다. 출처는 주석으로 남긴다.

이 순간부터 두 프롬프트 모두 우리 것이 되고, 바꾸려면 커밋이 필요해진다. 지금은
LangChain이 기본값을 바꾸면 조용히 따라간다.

## 4. 범위

### 2A — 재작성 (이 문서가 승인을 구하는 범위)

**포함**

- `ConversationalRetrievalChain` → §3.2의 LCEL 조립
- `ConversationBufferMemory` → `list[BaseMessage]`
- `CONDENSE_PROMPT`를 모듈 상수로 소유 (문구 그대로)
- LLM 호출 2회를 `ainvoke`로 전환
- `requirements.txt`에서 `langchain-classic` 관련 주석 정리
- `ARCHITECTURE.md`·`README.md`에서 "한시적 발판" 서술 제거
- `test_scoring_retriever.py`의 `ConversationalRetrievalChain.from_llm`
  몽키패치를 새 구조에 맞게 수정 (§7.1)

**동작 동등이 성공 기준이다.** §1.2의 발견 셋 중 무엇도 이 단계에서 고치지 않는다.

**제외**

- `ScoringRetriever`, `_format_sources`, `vector_store.py`, `chunking.py`,
  `document_processor.py` — 한 줄도 건드리지 않는다
- 프롬프트 문구 변경 (한국어 condense 포함)
- 재인덱싱 — 검색 계층이 그대로이므로 불필요하다

### 2B — 재작성이 열어주는 후속 작업 (별도 문서·커밋)

우선순위 순이며, 각각 독립적으로 판단·되돌리기 가능하다.

| 항목 | 근거 |
|---|---|
| ~~`[출처: 문서명]` 규칙 정리~~ | **완료(`2d49620`)** — 규칙을 빼는 대신 청크마다 `[출처: 문서명, p.N]` 헤더를 실어 규칙을 지킬 수 있게 했다. 청크당 한 줄만큼 토큰이 는다. |
| ~~조건부 OCR 주의문~~ | **완료(`c34e9ac`)** — 컨텍스트에 `ocr_block_prefix`가 있을 때만 붙는다. 마커가 있을 때의 프롬프트는 이전과 바이트 단위로 동일함을 확인했다. |
| ~~LLM 호출 재시도~~ | **완료(`3dee526`)** — `ConnectionError`와 `httpx.TimeoutException`만 재시도한다. `ollama.ResponseError`는 404와 500이 섞여 있어 제외. `llm_max_attempts`로 끌 수 있다. |
| ~~`max_tokens` 연결 또는 삭제~~ | **완료(`2d49620`)** — `ChatOllama(num_predict=...)`로 연결했다. |
| 한국어 condense 프롬프트 | **남음.** §3.1. 효과는 측정 대상이지 자명하지 않고, 정답 셋이 없어 A/B가 육안 판단이다. |

규칙 4번(없는 정보는 "찾을 수 없다"고 명시)과 5번(검색 0건이면 성실히 답변)의
경계는 **그대로 두기로 했다.** 모델 판단에 맡긴다.

**LangChain Agent Middleware는 여전히 범위 밖이다.** 재작성 후에도 이 파이프라인은
루프 없는 고정 2단계이며, `create_agent` 기반 훅을 걸 자리가 없다. 도입은 파이프라인을
에이전트로 바꾸겠다는 별개의 결정이다.

## 5. 설정

애플리케이션 설정에 **추가·제거되는 항목이 없다.** `max_tokens` 처리는 2B로 미룬다.

`requirements.txt`에서 `langchain-classic`은 **명시적 의존성이 아니므로 지울 줄이
없다.** `langchain-community` 0.4.2가 전이 의존성으로 계속 끌고 온다. 바뀌는 것은
그것이 *우리 코드가 쓰는 것*에서 *쓰지 않는 전이 패키지*가 된다는 사실이며,
주석과 문서에서 "한시적 발판" 서술을 걷어내는 것으로 반영한다.

## 6. 한계

- **동작 동등을 자동으로 증명할 수 없다.** 답변은 LLM이 만들고 온도가 0이어도
  프롬프트 조립이 한 글자만 달라지면 결과가 달라진다. §7.2가 조립 결과를 문자열
  수준에서 단언하는 이유이며, 그것이 이 작업에서 얻을 수 있는 최선이다.
- **`langchain-classic`은 완전히 사라지지 않는다.** 전이 의존성으로 남는다
  (§5). 없어지는 것은 우리 코드의 import뿐이다.
- **비동기 전환의 효과는 동시 요청에서만 나타난다.** 단일 사용자 기준 응답 시간은
  변하지 않는다. 과장하지 않는다.
- **`_format_sources`가 받는 문서 순서가 재배치된 상태라는 가정은 유지된다.**
  재작성이 이 결합을 없애지는 않는다. 검색과 출처 포매팅 사이에 단계가 끼면
  여전히 조용히 깨질 수 있다.
- **§1.2의 발견 셋은 이 단계에서 고쳐지지 않는다.** 특히 발견 1은 LLM이 문서명을
  지어낼 수 있는 상태를 그대로 둔다는 뜻이다. 2A가 끝나는 즉시 2B 첫 항목으로
  가야 한다.

## 7. 검증

### 7.1 기존 테스트가 계약이다

전체 168개가 통과해야 한다. 그중 **`ask_question` 경로를 직접 건드리는 것은
`test_scoring_retriever.py::test_ask_question_injects_the_reorder_setting_into_the_retriever`
하나뿐**이며, 이 테스트는 `ConversationalRetrievalChain.from_llm`을 몽키패치해
리트리버를 가로챈다. 체인이 사라지므로 이 대역은 새 구조에 맞춰 다시 써야 한다.

**이 테스트가 지키는 것(`RETRIEVAL_REORDER`가 리트리버까지 전달되는지, 양방향으로)은
그대로 지켜져야 한다.** 대역 방식만 바꾸고 단언은 유지한다.

나머지 167개는 수정 없이 통과해야 한다. 하나라도 손대야 한다면 그것은 동작이
바뀌었다는 신호이므로 멈추고 이유를 확인한다.

### 7.2 조립 결과를 문자열로 고정하는 새 테스트

재작성에서 실제로 깨지기 쉬운 것은 "체인이 도는가"가 아니라 **"LLM에게 정확히 같은
문자열이 가는가"** 이다. §3.1의 명세를 테스트로 옮긴다.

| 테스트 | 지키는 것 |
|---|---|
| 첫 질문 | `chat_history`가 비면 **condense LLM 호출이 일어나지 않고** 질문이 그대로 검색에 들어가는지 |
| 후속 질문 | condense 호출이 정확히 1회, 프롬프트의 `{chat_history}`가 `"\nHuman: …\nAssistant: …"` 형식인지 |
| 재작성 질문 전파 | `QA_PROMPT`의 `{question}`에 **원 질문이 아니라 재작성된 질문**이 들어가는지 (`rephrase_question=True` 재현) |
| 컨텍스트 조립 | `{context}`가 `page_content`를 `"\n\n"`로 이은 것과 정확히 같은지 — 메타데이터가 새어 들어가지 않는지 |
| 메모리 누적 | 한 턴 뒤 히스토리에 `HumanMessage`/`AIMessage`가 순서대로 쌓이는지, `clear_conversation` 후 비는지 |
| 대화 격리 | 서로 다른 `conversation_id`의 히스토리가 섞이지 않는지 |

LLM은 호출 인자를 기록하는 대역으로 세운다. 실제 Ollama를 부르지 않는다.

### 7.3 수동 A/B

재작성 전후로 **같은 문서·같은 질문**에 대해 답변을 비교한다. 온도가 0이므로
프롬프트가 동일하면 답변도 동일해야 한다. 다르면 §7.2가 놓친 조립 차이가 있다는
뜻이다.

- 첫 질문 (condense 없음)
- 후속 질문 (condense 있음) — 지시대명사를 포함해 재작성이 실제로 필요한 질문
- 문서 선택(`document_ids`)을 건 질문
- 검색 결과가 0건인 질문

### 7.4 비동기 확인

동시 요청 2건이 직렬화되지 않는지 확인한다. 재작성 전에는 두 요청의 총 소요가
합에 가깝고, 후에는 최댓값에 가까워야 한다. 정밀 측정이 목적이 아니라 **루프가
풀렸다는 사실 확인**이 목적이다.

## 8. 참고

- `langchain_classic/chains/conversational_retrieval/base.py` — §3.1 명세의 출처
- `langchain_classic/chains/conversational_retrieval/prompts.py` — 영어 condense 기본 프롬프트
- `langchain_classic/chains/combine_documents/stuff.py` — `document_separator="\n\n"`, `DEFAULT_DOCUMENT_PROMPT="{page_content}"`
- 커밋 `8f892f7` — `langchain-classic`을 발판으로 도입한 1단계
