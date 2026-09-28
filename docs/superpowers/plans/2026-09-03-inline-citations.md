# 인라인 각주 출처 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 답변 본문의 `[1]` 표시를 각주로 바꾸고, 마우스를 올리면 해당 청크를 미리 보여준다.

**Architecture:** 리랭킹 직후·재배치 직전에 `citation_index`를 메타데이터에 심어, 컨텍스트 헤더와 출처 응답이 **같은 번호로 같은 청크**를 가리키게 한다. 프론트엔드는 마크다운 렌더 결과의 텍스트에서 `[숫자]`를 찾아 각주로 치환하고, 번호로 `sources` 배열을 조회한다(배열 위치가 아니라 `citation_index` 필드로 찾는다).

**Tech Stack:** Python 3.12, langchain-core 1.6.1, pytest 7.4.4, React 18 / TypeScript 5.3, Vite 5, **vitest (신규 도입 — Task 5)**

**설계 문서:** [2026-09-03-inline-citations-design.md](../specs/2026-09-03-inline-citations-design.md)

---

## 사전 준비

**Python은 반드시 프로젝트 venv로 실행한다.** 시스템 `python`에는 의존성이 없다.

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System/backend
./app/venv/bin/python -m pytest -q
```

착수 전 기준선: **200 passed**. 다르면 멈추고 원인을 확인한다.

**⚠️ 개행 문자.** `rag_service.py`와 `responses.py`는 **CRLF**, `backend/tests/*.py`와 `frontend/src/**`는 **LF**다. 통짜로 다시 쓰지 말고 부분 편집한다. 커밋 전 확인:

```bash
cd backend && for f in app/services/rag_service.py app/api/models/responses.py; do echo "$f: $(grep -c $'\r$' $f) / $(wc -l < $f)"; done
```

각 쌍이 같아야 한다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `backend/app/services/rag_service.py` | `citation_index` 부여, 헤더 형식, 프롬프트, 응답 전달 |
| `backend/app/api/models/responses.py` | `SourceDocument.citation_index` |
| `frontend/src/utils/citations.ts` (신규) | **순수 함수** — 텍스트를 각주 조각으로 쪼갠다. React 의존 없음 |
| `frontend/src/utils/citations.test.ts` (신규) | 위 함수의 단위 테스트 |
| `frontend/src/components/CitationFootnote.tsx` (신규) | 각주 + hover 카드 |
| `frontend/src/components/Message.tsx` | `ReactMarkdown`에 각주 렌더러 연결 |
| `frontend/src/types/api.types.ts` | `citation_index` 필드 |

**파싱을 `citations.ts`로 떼어내는 이유:** `[숫자]` 처리가 이 작업에서 가장 틀리기 쉬운 부분인데, React 컴포넌트 안에 두면 렌더링 없이는 검증할 수 없다. 순수 함수로 두면 입력·출력만으로 테스트된다.

---

## Task 1: `citation_index` 부여

**Files:**
- Modify: `backend/app/services/rag_service.py`
- Test: `backend/tests/test_scoring_retriever.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_scoring_retriever.py` 끝에 추가:

```python
def test_citation_index_follows_score_order_not_context_order():
    """번호는 출처 목록 순서(점수순)와 같아야 한다.

    재배치를 켜면 리트리버가 돌려주는 순서는 U자로 흐트러진다. 그 순서대로
    번호를 매기면 사용자가 보는 sources[0]과 [1]이 다른 청크가 된다.
    """
    store = FakeVectorStore(
        _docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)])
    )
    retriever = ScoringRetriever(
        vector_store=store, k=4, rerank=True, candidate_k=4,
        reranker=FakeReranker(scores=[0.1, 0.4, 0.3, 0.2]), reorder=True,
    )

    docs = retriever.invoke("질문")

    # 리랭크 점수 내림차순: b(0.4) c(0.3) d(0.2) a(0.1)
    by_index = {d.metadata["citation_index"]: d.page_content for d in docs}
    assert by_index == {1: "b", 2: "c", 3: "d", 4: "a"}


def test_citation_index_survives_the_reordering():
    """재배치는 순서만 바꾼다. 번호는 문서를 따라가야 한다."""
    store = FakeVectorStore(
        _docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)])
    )
    retriever = ScoringRetriever(
        vector_store=store, k=4, rerank=True, candidate_k=4,
        reranker=FakeReranker(scores=[0.1, 0.4, 0.3, 0.2]), reorder=True,
    )

    docs = retriever.invoke("질문")

    assert all("citation_index" in d.metadata for d in docs)
    assert sorted(d.metadata["citation_index"] for d in docs) == [1, 2, 3, 4]


def test_citation_index_is_assigned_without_reranking():
    """리랭커를 꺼도 번호는 붙는다. 그때 기준은 벡터 점수 순서다."""
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8)]))
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=False, reorder=False,
    )

    docs = retriever.invoke("질문")

    assert [d.metadata["citation_index"] for d in docs] == [1, 2]
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_scoring_retriever.py -k citation_index -v
```

기대: 3개 FAIL — `KeyError: 'citation_index'`

- [ ] **Step 3: 번호 부여를 구현한다**

`backend/app/services/rag_service.py`의 `_get_relevant_documents`에서 다음 부분을 찾는다:

```python
        docs = [doc for doc, _ in results]
        if self.rerank:
            docs = self._rerank(query, docs)

        if not self.reorder:
            return docs
```

교체한다:

```python
        docs = [doc for doc, _ in results]
        if self.rerank:
            docs = self._rerank(query, docs)
        else:
            docs = docs[: self.k]

        # 번호는 반드시 재배치 "전"에, 그리고 _format_sources가 쓸 순서 그대로
        # 매긴다. 재배치 후에 매기면 U자로 흐트러진 위치의 번호가 되어,
        # 사용자가 보는 sources[0]과 [1]이 다른 청크를 가리키게 된다.
        for number, doc in enumerate(docs, start=1):
            doc.metadata["citation_index"] = number

        if not self.reorder:
            return docs
```

> `else: docs = docs[: self.k]` 를 더한 이유: 리랭킹을 끄면 지금까지는 Chroma가
> 이미 `k`개만 돌려줬으므로 잘라낼 필요가 없었다. 번호를 매기게 되면서 길이를
> 명시적으로 맞춰 두는 편이 안전하다. `fetch_k`가 `self.k`이므로 동작은 같다.

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_scoring_retriever.py -v
```

기대: 기존 17개 + 신규 3개 = 20 passed

```bash
./app/venv/bin/python -m pytest -q
```

기대: **203 passed**

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/services/rag_service.py backend/tests/test_scoring_retriever.py
git commit -m "feat: number retrieved chunks before reordering"
```

---

## Task 2: 컨텍스트 헤더와 프롬프트

**Files:**
- Modify: `backend/app/services/rag_service.py`
- Test: `backend/tests/test_rag_chain.py`, `backend/tests/test_local_rag_defaults.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_rag_chain.py`에서 기존 `test_each_chunk_carries_a_source_header`와
`test_a_chunk_without_a_page_gets_a_header_without_one` **두 함수를 삭제하고**, 그
자리에 아래를 넣는다. (헤더 형식이 바뀌므로 기존 단언은 더 이상 유효하지 않다.)

```python
def test_each_chunk_carries_a_numbered_header(monkeypatch):
    """번호 인용이 가능하려면 컨텍스트에 번호가 있어야 한다.

    문서명도 함께 둔다. 모델이 어느 문서를 읽는지 판단하는 데 쓰인다.
    """
    llm = RecordingLLM()
    results = [
        (Document(page_content="첫 청크",
                  metadata={"filename": "a.pdf", "page": 3, "citation_index": 1}), 0.9),
        (Document(page_content="둘째 청크",
                  metadata={"filename": "b.pdf", "page": 7, "citation_index": 2}), 0.5),
    ]
    service = build_service(monkeypatch, llm, results)

    asyncio.run(service.ask_question("질문"))

    answer_prompt = llm.prompts("answer")[0]
    assert "[1] a.pdf p.3\n첫 청크" in answer_prompt
    assert "[2] b.pdf p.7\n둘째 청크" in answer_prompt
    assert "첫 청크\n\n[2] b.pdf" in answer_prompt


def test_a_chunk_without_a_page_gets_a_header_without_one(monkeypatch):
    """페이지는 PDF에만 있다. TXT/DOCX에서 'p.None'이 나오면 안 된다."""
    llm = RecordingLLM()
    results = [
        (Document(page_content="본문",
                  metadata={"filename": "메모.txt", "citation_index": 1}), 0.9)
    ]
    service = build_service(monkeypatch, llm, results)

    asyncio.run(service.ask_question("질문"))

    answer_prompt = llm.prompts("answer")[0]
    assert "[1] 메모.txt\n본문" in answer_prompt
    assert "p.None" not in answer_prompt
```

`backend/tests/test_local_rag_defaults.py`에서 `test_qa_prompt_warns_about_ocr_marker`
**바로 아래**에 추가:

```python
def test_qa_prompt_asks_for_numbered_citations():
    """번호 인용을 요구해야 프론트엔드가 각주로 바꿀 수 있다."""
    from app.services.rag_service import QA_PROMPT

    assert "[1]" in QA_PROMPT.template
    assert "[출처: 문서명]" not in QA_PROMPT.template
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_rag_chain.py -k header -v
./app/venv/bin/python -m pytest tests/test_local_rag_defaults.py -k numbered -v
```

기대: 모두 FAIL — 헤더가 아직 `[출처: a.pdf, p.3]` 형식이고, 프롬프트에 `[1]`이 없다

- [ ] **Step 3: 헤더와 프롬프트를 바꾼다**

3a. `_format_document`를 교체한다. 기존:

```python
def _format_document(doc: Document) -> str:
    """청크 하나를 출처 헤더 + 본문으로 만든다.

    페이지는 PDF에만 있으므로(TXT/DOCX는 None) 있을 때만 붙인다. 문서명이
    없는 경우의 "Unknown"은 _format_sources가 쓰는 값과 맞춘 것이다.
    """
    filename = doc.metadata.get("filename", "Unknown")
    page = doc.metadata.get("page")
    header = f"[출처: {filename}, p.{page}]" if page is not None else f"[출처: {filename}]"
    return f"{header}\n{doc.page_content}"
```

교체 후:

```python
def _format_document(doc: Document) -> str:
    """청크 하나를 번호 헤더 + 본문으로 만든다.

    번호는 모델이 인용할 대상이고, 문서명과 쪽은 모델이 지금 어느 문서를 읽고
    있는지 판단하는 데 쓴다. 둘 다 필요하다.

    페이지는 PDF에만 있으므로(TXT/DOCX는 None) 있을 때만 붙인다. 문서명이
    없는 경우의 "Unknown"은 _format_sources가 쓰는 값과 맞춘 것이다.
    """
    number = doc.metadata.get("citation_index")
    filename = doc.metadata.get("filename", "Unknown")
    page = doc.metadata.get("page")

    label = f"{filename} p.{page}" if page is not None else filename
    header = f"[{number}] {label}" if number is not None else f"[출처: {label}]"
    return f"{header}\n{doc.page_content}"
```

> 번호가 없는 문서(리트리버를 거치지 않은 경우)는 예전 형식으로 물러난다.
> 컨텍스트에 `[None]`이 들어가는 것보다 낫다.

3b. `QA_PROMPT`에서 규칙 2번 줄을 찾는다:

```
2. **출처 인용**: 답변에 사용한 정보의 출처를 [출처: 문서명] 형식으로 표시하세요.
```

교체한다:

```
2. **출처 인용**: 문장 끝에 근거가 된 문서의 번호를 [1] 형식으로 붙이세요. 여러 개면 [1][3]처럼 이어 씁니다. 컨텍스트에 없는 번호는 절대 쓰지 마세요.
```

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest -q
```

기대: **204 passed** (203 + `test_qa_prompt_asks_for_numbered_citations` 1개.
`test_rag_chain.py`는 기존 2개를 교체했으므로 개수 변화 없음)

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/services/rag_service.py backend/tests/test_rag_chain.py backend/tests/test_local_rag_defaults.py
git commit -m "feat: number the context headers and ask the model to cite by number"
```

---

## Task 3: 응답에 번호 싣기

**Files:**
- Modify: `backend/app/api/models/responses.py`, `backend/app/services/rag_service.py`
- Test: `backend/tests/test_source_formatting.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_source_formatting.py` 끝에 추가:

```python
def test_citation_index_reaches_the_response():
    """프론트엔드는 배열 위치가 아니라 이 필드로 각주를 찾는다."""
    sources = _format([
        Document(page_content="a", metadata={"similarity_score": 0.5, "citation_index": 2}),
        Document(page_content="b", metadata={"similarity_score": 0.9, "citation_index": 1}),
    ])

    # 점수순으로 정렬되므로 b가 앞이고, 번호는 각자 따라온다.
    assert [s.content for s in sources] == ["b", "a"]
    assert [s.citation_index for s in sources] == [1, 2]


def test_missing_citation_index_is_none_not_a_crash():
    """리트리버를 거치지 않은 문서도 _format_sources로 들어온다."""
    sources = _format([
        Document(page_content="a", metadata={"similarity_score": 0.9}),
    ])

    assert sources[0].citation_index is None
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_source_formatting.py -k citation -v
```

기대: FAIL — `SourceDocument`에 `citation_index` 필드가 없다

- [ ] **Step 3: 필드를 추가한다**

3a. `backend/app/api/models/responses.py`의 `SourceDocument`에서 `rerank_score` 줄
**바로 아래**에 추가:

```python
    citation_index: Optional[int] = Field(None, description="1-based number used for inline footnotes")
```

3b. `backend/app/services/rag_service.py`의 `_format_sources`에서
`rerank_score=doc.metadata.get("rerank_score"),` 줄 **바로 아래**에 추가:

```python
                citation_index=doc.metadata.get("citation_index"),
```

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest -q
```

기대: **206 passed**

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/api/models/responses.py backend/app/services/rag_service.py backend/tests/test_source_formatting.py
git commit -m "feat: expose the citation number in the chat response"
```

---

## Task 4: 프론트엔드 타입

**Files:**
- Modify: `frontend/src/types/api.types.ts`

- [ ] **Step 1: 필드를 추가한다**

`frontend/src/types/api.types.ts`의 `SourceDocument`에서 `rerank_score: number | null;`
**바로 아래**에 추가:

```typescript
  citation_index: number | null;
```

- [ ] **Step 2: 타입 검사**

```bash
cd frontend && npx tsc --noEmit
```

기대: 출력 없음

- [ ] **Step 3: 커밋한다**

```bash
cd .. && git add frontend/src/types/api.types.ts
git commit -m "feat: add citation_index to the source type"
```

---

## Task 5: 프론트엔드 테스트 도구 도입

이 저장소에는 프론트엔드 테스트 도구가 **하나도 없다**(`package.json`에 `test`
스크립트도, vitest·jest·testing-library도 없음). 설계 문서 §7.2가 요구하는 테스트를
돌리려면 먼저 도구가 있어야 한다.

Vite 프로젝트이므로 vitest가 자연스러운 선택이다. 이 태스크는 **순수 함수 테스트에
필요한 최소한만** 넣는다 — DOM 렌더링 테스트는 하지 않으므로 jsdom과
testing-library는 넣지 않는다.

- [ ] **Step 1: vitest를 설치한다**

```bash
cd frontend && npm install -D vitest@2.1.8
```

- [ ] **Step 2: 스크립트를 추가한다**

`frontend/package.json`의 `"scripts"`에서 `"lint"` 줄 **바로 위**에 추가:

```json
    "test": "vitest run",
```

- [ ] **Step 3: 동작을 확인한다**

임시 파일 `frontend/src/utils/smoke.test.ts`를 만든다:

```typescript
import { describe, it, expect } from 'vitest';

describe('vitest', () => {
  it('runs', () => {
    expect(1 + 1).toBe(2);
  });
});
```

```bash
npm test
```

기대: `1 passed`

그리고 임시 파일을 지운다:

```bash
rm src/utils/smoke.test.ts
```

- [ ] **Step 4: 커밋한다**

```bash
cd .. && git add frontend/package.json frontend/package-lock.json
git commit -m "chore: add vitest so frontend logic can be tested"
```

---

## Task 6: 각주 파싱 (순수 함수)

**Files:**
- Create: `frontend/src/utils/citations.ts`
- Test: `frontend/src/utils/citations.test.ts`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`frontend/src/utils/citations.test.ts` 신규 생성:

```typescript
import { describe, it, expect } from 'vitest';
import { splitCitations, buildCitationMap } from './citations';
import type { SourceDocument } from '../types/api.types';

describe('splitCitations', () => {
  it('유효한 번호를 각주 조각으로 분리한다', () => {
    expect(splitCitations('예산은 3,200억 원이다[1].', [1, 2])).toEqual([
      { type: 'text', value: '예산은 3,200억 원이다' },
      { type: 'citation', value: '[1]', number: 1 },
      { type: 'text', value: '.' },
    ]);
  });

  it('없는 번호는 원문 그대로 둔다', () => {
    // 존재하지 않는 근거를 링크처럼 보이게 하는 것이 더 나쁘다.
    expect(splitCitations('근거가 있다[7].', [1, 2])).toEqual([
      { type: 'text', value: '근거가 있다[7].' },
    ]);
  });

  it('연속된 번호를 각각 분리한다', () => {
    const parts = splitCitations('두 곳에 나온다[1][3].', [1, 2, 3]);
    expect(parts.filter((p) => p.type === 'citation')).toEqual([
      { type: 'citation', value: '[1]', number: 1 },
      { type: 'citation', value: '[3]', number: 3 },
    ]);
  });

  it('번호가 없으면 통째로 텍스트다', () => {
    expect(splitCitations('각주 없는 문장.', [1])).toEqual([
      { type: 'text', value: '각주 없는 문장.' },
    ]);
  });

  it('빈 문자열을 견딘다', () => {
    expect(splitCitations('', [1])).toEqual([]);
  });

  it('숫자가 아닌 대괄호는 건드리지 않는다', () => {
    expect(splitCitations('[출처: a.pdf] 참고', [1])).toEqual([
      { type: 'text', value: '[출처: a.pdf] 참고' },
    ]);
  });
});

describe('buildCitationMap', () => {
  const make = (n: number | null, content: string) =>
    ({ citation_index: n, content } as SourceDocument);

  it('배열 위치가 아니라 citation_index로 찾는다', () => {
    // 일부러 번호와 배열 순서를 어긋나게 둔다. 위치로 찾는 구현이면 실패한다.
    const map = buildCitationMap([make(2, 'b'), make(1, 'a')]);

    expect(map.get(1)?.content).toBe('a');
    expect(map.get(2)?.content).toBe('b');
  });

  it('번호가 없는 출처는 제외한다', () => {
    const map = buildCitationMap([make(null, 'x'), make(1, 'a')]);

    expect(map.size).toBe(1);
    expect(map.get(1)?.content).toBe('a');
  });
});
```

- [ ] **Step 2: 실패를 확인한다**

```bash
cd frontend && npm test
```

기대: FAIL — `Failed to resolve import "./citations"`

- [ ] **Step 3: 구현한다**

`frontend/src/utils/citations.ts` 신규 생성:

```typescript
/**
 * 답변 텍스트에서 [숫자] 인용을 찾아 조각으로 나눈다.
 *
 * React를 쓰지 않는 순수 함수로 둔 이유: [숫자] 처리가 이 기능에서 가장
 * 틀리기 쉬운 부분인데, 컴포넌트 안에 있으면 렌더링 없이는 검증할 수 없다.
 */

import type { SourceDocument } from '../types/api.types';

export type CitationPart =
  | { type: 'text'; value: string }
  | { type: 'citation'; value: string; number: number };

const CITATION_PATTERN = /\[(\d+)\]/g;

/**
 * `available`에 있는 번호만 각주로 분리한다. 없는 번호는 텍스트로 남는다 —
 * 존재하지 않는 근거를 링크처럼 보이게 하는 것이 더 나쁘다.
 */
export function splitCitations(
  text: string,
  available: number[]
): CitationPart[] {
  if (!text) return [];

  const valid = new Set(available);
  const parts: CitationPart[] = [];
  let cursor = 0;

  for (const match of text.matchAll(CITATION_PATTERN)) {
    const number = Number(match[1]);
    if (!valid.has(number)) continue;

    const start = match.index!;
    if (start > cursor) {
      parts.push({ type: 'text', value: text.slice(cursor, start) });
    }
    parts.push({ type: 'citation', value: match[0], number });
    cursor = start + match[0].length;
  }

  if (cursor < text.length) {
    parts.push({ type: 'text', value: text.slice(cursor) });
  }
  return parts;
}

/**
 * 번호 → 출처. 배열 위치로 찾으면 정렬이 바뀌는 순간 틀어지므로
 * citation_index를 키로 쓴다. 번호가 없는 출처는 각주 대상이 아니다.
 */
export function buildCitationMap(
  sources: SourceDocument[]
): Map<number, SourceDocument> {
  const map = new Map<number, SourceDocument>();
  for (const source of sources) {
    if (source.citation_index != null) {
      map.set(source.citation_index, source);
    }
  }
  return map;
}
```

- [ ] **Step 4: 통과를 확인한다**

```bash
npm test
```

기대: 8 passed

```bash
npx tsc --noEmit
```

기대: 출력 없음

- [ ] **Step 5: 커밋한다**

```bash
cd .. && git add frontend/src/utils/citations.ts frontend/src/utils/citations.test.ts
git commit -m "feat: split answer text into citation parts"
```

---

## Task 7: 각주 컴포넌트와 hover 카드

**Files:**
- Create: `frontend/src/components/CitationFootnote.tsx`
- Create: `frontend/src/styles/CitationFootnote.css`

- [ ] **Step 1: 컴포넌트를 만든다**

`frontend/src/components/CitationFootnote.tsx` 신규 생성:

```tsx
import React, { useState } from 'react';
import { SourceDocument } from '../types/api.types';
import '../styles/CitationFootnote.css';

interface CitationFootnoteProps {
  source: SourceDocument;
}

/**
 * 각주 숫자와 hover 카드.
 *
 * 데스크톱에서는 CSS hover로 열리고, 마우스가 없는 환경을 위해 탭으로도
 * 토글된다(open 상태).
 */
export const CitationFootnote: React.FC<CitationFootnoteProps> = ({ source }) => {
  const [open, setOpen] = useState(false);
  const thumbnail = source.image_urls[0];

  return (
    <span
      className={`citation${open ? ' open' : ''}`}
      onClick={() => setOpen(!open)}
    >
      <sup className="citation-number">{source.citation_index}</sup>

      <span className="citation-card">
        <span className="citation-title">{source.document_name}</span>
        <span className="citation-meta">
          {source.page != null && `${source.page}쪽 · `}청크 #{source.chunk_index}
        </span>
        <span className="citation-body">{source.content}</span>

        {thumbnail && (
          <img
            className="citation-thumb"
            src={thumbnail}
            alt="문서 도표"
            loading="lazy"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
        )}

        {source.similarity_score != null && (
          <span className="citation-scores">
            Relevance {(source.similarity_score * 100).toFixed(1)}%
            {source.rerank_score != null &&
              ` · Rerank ${(source.rerank_score * 100).toFixed(1)}%`}
          </span>
        )}
      </span>
    </span>
  );
};
```

- [ ] **Step 2: 스타일을 만든다**

`frontend/src/styles/CitationFootnote.css` 신규 생성:

```css
.citation {
  position: relative;
  cursor: help;
}

.citation-number {
  color: #4a6fa5;
  font-weight: 700;
  font-size: 0.7em;
  vertical-align: super;
  margin: 0 1px;
}

.citation:hover .citation-number,
.citation.open .citation-number {
  text-decoration: underline;
}

.citation-card {
  visibility: hidden;
  opacity: 0;
  transition: opacity 0.12s;
  position: absolute;
  top: 150%;
  left: 50%;
  transform: translateX(-50%);
  width: 330px;
  max-width: 80vw;
  background: #1f2430;
  color: #e6e9ef;
  text-align: left;
  border-radius: 10px;
  padding: 12px 14px;
  font-size: 12.5px;
  line-height: 1.62;
  font-weight: 400;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.34);
  z-index: 40;
}

.citation:hover .citation-card,
.citation.open .citation-card {
  visibility: visible;
  opacity: 1;
}

.citation-title {
  display: block;
  font-weight: 700;
  color: #fff;
}

.citation-meta {
  display: block;
  color: #9aa4b5;
  font-size: 11.5px;
  margin-bottom: 8px;
}

/* 발췌는 3줄에서 자른다. 전체는 하단 출처 목록에서 볼 수 있다. */
.citation-body {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  color: #cbd3e1;
  border-left: 2px solid #4a5568;
  padding-left: 9px;
}

.citation-thumb {
  display: block;
  margin-top: 9px;
  width: 100%;
  border-radius: 5px;
}

.citation-scores {
  display: block;
  margin-top: 9px;
  color: #7f8b9e;
  font-size: 11px;
}
```

- [ ] **Step 3: 타입 검사**

```bash
cd frontend && npx tsc --noEmit
```

기대: 출력 없음

- [ ] **Step 4: 커밋한다**

```bash
cd .. && git add frontend/src/components/CitationFootnote.tsx frontend/src/styles/CitationFootnote.css
git commit -m "feat: add the citation footnote with a hover preview card"
```

---

## Task 8: 답변 본문에 각주 연결

**Files:**
- Modify: `frontend/src/components/Message.tsx`

- [ ] **Step 1: 각주 렌더러를 붙인다**

`frontend/src/components/Message.tsx`의 임포트 블록에서
`import { SourceCitation } from './SourceCitation';` **바로 아래**에 추가:

```tsx
import { CitationFootnote } from './CitationFootnote';
import { splitCitations, buildCitationMap } from '../utils/citations';
```

컴포넌트 본문에서 `const isUser = message.role === 'user';` **바로 아래**에 추가:

```tsx
  // 마크다운이 만든 텍스트 노드에서 [숫자]를 찾아 각주로 바꾼다. 번호가 없는
  // 출처(리트리버를 거치지 않은 경우)는 각주 대상에서 빠진다.
  const citations = buildCitationMap(message.sources ?? []);
  const available = [...citations.keys()];

  const renderWithCitations = (children: React.ReactNode): React.ReactNode =>
    React.Children.map(children, (child) => {
      if (typeof child !== 'string') return child;
      return splitCitations(child, available).map((part, i) =>
        part.type === 'text' ? (
          part.value
        ) : (
          <CitationFootnote key={i} source={citations.get(part.number)!} />
        )
      );
    });
```

그리고 `<ReactMarkdown>{message.content}</ReactMarkdown>` 줄을 교체한다:

```tsx
            <ReactMarkdown
              components={{
                p: ({ children }) => <p>{renderWithCitations(children)}</p>,
                li: ({ children }) => <li>{renderWithCitations(children)}</li>,
              }}
            >
              {message.content}
            </ReactMarkdown>
```

> `p`와 `li`만 가공한다. `code`는 건드리지 않으므로 코드 블록 안의 `[1]`은
> 그대로 남는다.

- [ ] **Step 2: 타입 검사와 테스트**

```bash
cd frontend && npx tsc --noEmit && npm test
```

기대: tsc 출력 없음, 8 passed

- [ ] **Step 3: 커밋한다**

```bash
cd .. && git add frontend/src/components/Message.tsx
git commit -m "feat: render inline citation footnotes in the answer"
```

---

## Task 9: 수동 확인 (머지 전 필수)

설계 문서 §7.3이 요구하는 확인이다. **§7.2 표의 "도표 없는 청크" 행도 여기서
확인한다** — DOM 렌더링 테스트 도구(jsdom·testing-library)를 일부러 넣지 않았으므로
컴포넌트 분기는 눈으로 본다. **모델이 실제로 번호 인용을 쓰는지는 코드로
증명할 수 없다.**

- [ ] **Step 1: 전체 테스트**

```bash
cd backend && ./app/venv/bin/python -m pytest -q
cd ../frontend && npx tsc --noEmit && npm test
```

기대: 백엔드 실패 0, tsc 출력 없음, 프론트 8 passed

- [ ] **Step 2: 서버를 띄운다**

```bash
cd backend && ./app/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

다른 터미널에서:

```bash
cd frontend && npm run dev
```

- [ ] **Step 3: 확인한다**

문서를 하나 올리고 질문을 3~4개 던져 다음을 본다.

- **모델이 `[1]` 형식으로 인용하는가** — 이것이 실패하면 각주가 하나도 안 생긴다.
  그때는 `QA_PROMPT` 규칙 2번 문구를 조정하고 다시 확인한다.
- 각주에 마우스를 올렸을 때 **카드가 본문을 가리지 않고 화면 밖으로 잘리지 않는가**
- 카드의 문서명·쪽·발췌가 **하단 목록의 같은 번호 항목과 일치하는가**
  (§3.1의 어긋남이 실제로 없는지 눈으로 확인하는 지점이다)
- 도표가 있는 청크에서 썸네일이 뜨고, 없는 청크에서는 그 자리가 사라지는가
- 인용이 하나도 없는 답변이 이전과 똑같이 보이는가

- [ ] **Step 4: 관찰을 기록한다**

`docs/troubleshooting/2026-09-03-inline-citations.md`를 만들어 위 항목의 결과를
남긴다. 특히 **모델이 인용을 얼마나 안정적으로 하는지**를 적는다 — 이 기능의
효용이 거기에 달려 있다.

- [ ] **Step 5: 커밋한다**

```bash
git add docs/troubleshooting/2026-09-03-inline-citations.md
git commit -m "docs: record the inline citation manual check"
```
