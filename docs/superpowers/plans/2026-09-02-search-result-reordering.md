# 검색 결과 재배치 (Lost in the Middle 완화) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 벡터 검색이 돌려준 청크를 컨텍스트에 넣기 전에 재배치해, 관련성 높은 청크가 LLM이 취약한 컨텍스트 중앙 구간에 묻히지 않게 한다.

**Architecture:** `ScoringRetriever._get_relevant_documents`가 유사도 점수를 메타데이터에 심은 **뒤**, `langchain_community`의 `LongContextReorder`로 순서만 바꿔 반환한다. 재배치는 LLM을 위한 내부 최적화이므로 `RAGService._format_sources`가 사용자에게 보이는 출처는 다시 점수 내림차순으로 되돌린다. 검색 단계만 바뀌므로 재인덱싱은 필요 없다.

**Tech Stack:** Python 3.12, LangChain 0.1.20 / langchain-community 0.0.38, pytest, pydantic-settings

**설계 문서:** [2026-09-02-search-result-reordering-design.md](../specs/2026-09-02-search-result-reordering-design.md)

---

## 사전 지식 (이 저장소를 처음 보는 사람을 위해)

**Python 실행은 반드시 프로젝트 venv로 한다.** 시스템 `python`에는 langchain이 없다.

```bash
cd backend && app/venv/bin/python -m pytest tests/ -q
```

전체 스위트는 임베딩 모델을 로딩하지 않는 테스트만으로도 **약 23초**가 걸린다. 기준선은 **156 passed**이다.

**`backend/app/services/rag_service.py`는 이 저장소에서 유일하게 CRLF(`\r\n`) 줄바꿈을 쓴다.** `.gitattributes`가 없어 자동 정규화도 없다. 파일 전체를 다시 쓰는 방식(Python `open(...,"w")`, `cat > file` 등)으로 편집하면 줄바꿈이 LF로 바뀌면서 **파일 전체가 diff에 뜬다.** 반드시 Edit 도구로 부분 수정하고, 각 커밋 전에 `git diff --numstat`로 변경 줄 수를 확인한다.

**테스트에서 `RAGService`를 직접 생성하지 않는다.** `__init__`이 임베딩 모델 로딩과 Ollama 연결을 요구한다. 기존 테스트는 `RAGService.__new__(RAGService)`로 우회한다 (`tests/test_source_formatting.py:11`).

---

## 파일 구조

| 파일 | 역할 | 변경 |
|---|---|---|
| `backend/app/config.py` | 설정 | `retrieval_reorder` 추가 |
| `backend/app/services/rag_service.py` | 프롬프트·리트리버·RAG 서비스 | 재배치 적용, 출처 정렬, 프롬프트 문구 |
| `backend/.env.example` | 설정 문서 | `RETRIEVAL_REORDER` 추가 |
| `README.md` | 설정 표(191행 부근, **영문**) | `RETRIEVAL_REORDER` 행 추가 |
| `backend/tests/test_long_context_reorder.py` | **신규** — 라이브러리 계약 고정 | 생성 |
| `backend/tests/test_scoring_retriever.py` | 리트리버 동작 | 재배치 테스트 추가, 기존 테스트 1개 수정 |
| `backend/tests/test_source_formatting.py` | 출처 포매팅 | 점수순 정렬 테스트 추가 |
| `backend/tests/test_local_rag_defaults.py` | 기본값·프롬프트 | 설정 기본값·프롬프트 문구 테스트 추가 |

새 모듈은 만들지 않는다. 알고리즘을 라이브러리에서 가져오므로 소유할 코드가 없다.

---

## Task 1: 프로토타입 되돌리기

브레인스토밍 이전에 직접 구현한 `reorder_documents` 함수가 워킹트리에 남아 있다. 설계에서 라이브러리를 쓰기로 했으므로 삭제하고, `rag_service.py`를 CRLF 상태인 HEAD로 되돌린 뒤 시작한다.

**Files:**
- Revert 통째로: `backend/app/services/rag_service.py`
- Delete: `backend/tests/test_document_reorder.py`
- **부분 제거만**: `backend/app/config.py`, `backend/.env.example` (아래 경고 참조)

> **경고 — `git checkout --`를 config.py와 .env.example에 쓰지 말 것.**
> 두 파일에는 **다른 작업(청킹 개선)의 미커밋 변경**이 섞여 있다.
> 통째로 되돌리면 `semantic_chunker_min_chunk_chars`, `CHUNKING_STRATEGY`,
> `SEMANTIC_CHUNKER_*`, `RETRIEVAL_K=4→10`, `MAX_DOCUMENTS`가 전부 사라진다.
> 이 파일들은 **프로토타입이 넣은 줄만 골라서** 지운다.
> `rag_service.py`는 프로토타입 이전에 수정된 적이 없으므로 통째로 되돌려도 안전하다.

- [ ] **Step 1: 되돌릴 대상 확인**

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System
git diff --numstat backend/app/services/rag_service.py backend/app/config.py backend/.env.example
```

기대: `rag_service.py`가 240/196처럼 큰 수 — 줄바꿈이 통째로 LF로 바뀐 상태다.
`config.py`는 6/0, `.env.example`은 11/2 내외이며 **이 중 대부분은 다른 작업의 것이다.**

- [ ] **Step 2: `rag_service.py`만 통째로 되돌리기**

```bash
git checkout -- backend/app/services/rag_service.py
rm -f backend/tests/test_document_reorder.py
```

- [ ] **Step 3: `config.py`에서 프로토타입이 넣은 3줄만 제거**

```bash
python3 - <<'EOF'
p = "backend/app/config.py"
s = open(p, encoding="utf-8").read()
block = """    # 검색 결과를 관련성 높은 순서 그대로 넣지 않고, 1등을 맨 앞 / 2등을 맨 뒤로
    # 번갈아 배치해 중요한 청크가 컨텍스트 가운데에 묻히지 않게 한다.
    retrieval_reorder: bool = True
"""
assert block in s, "프로토타입 블록을 찾지 못했다 - 수동 확인 필요"
open(p, "w", encoding="utf-8").write(s.replace(block, "", 1))
print("config.py 정리 완료")
EOF
```

- [ ] **Step 4: `.env.example`에서 프로토타입이 넣은 1줄만 제거**

```bash
python3 - <<'EOF'
p = "backend/.env.example"
s = open(p, encoding="utf-8").read()
line = "RETRIEVAL_REORDER=true  # 관련성 높은 청크를 컨텍스트 앞/뒤 끝에 번갈아 배치 (Lost in the Middle 완화)\n"
assert line in s, "프로토타입 줄을 찾지 못했다 - 수동 확인 필요"
open(p, "w", encoding="utf-8").write(s.replace(line, "", 1))
print(".env.example 정리 완료")
EOF
```

- [ ] **Step 5: 다른 작업의 변경이 살아 있는지 확인**

```bash
grep -n "semantic_chunker_min_chunk_chars" backend/app/config.py
grep -n "CHUNKING_STRATEGY\|SEMANTIC_CHUNKER_\|MAX_DOCUMENTS\|RETRIEVAL_K=10" backend/.env.example
grep -c "RETRIEVAL_REORDER\|retrieval_reorder" backend/app/config.py backend/.env.example
```

기대: 앞의 두 명령은 결과가 나오고(다른 작업의 변경이 보존됨), 마지막 명령은 두 파일 모두 **`0`**이다.
하나라도 어긋나면 **멈추고 보고한다.**

- [ ] **Step 6: CRLF가 돌아왔는지 확인**

```bash
file backend/app/services/rag_service.py
```

기대: `... with CRLF line terminators`

- [ ] **Step 7: 워킹트리 상태 확인**

```bash
git status --short backend/
```

기대: `backend/` 아래에 `M backend/.env.example`, `M backend/app/config.py`,
`M backend/app/services/chunking.py`, `M backend/app/services/document_processor.py`,
`M backend/app/services/mineru_client.py`, `M backend/tests/test_chunking.py`,
`M backend/tests/test_mineru_client.py`가 남는다. **이들은 다른 작업의 변경분이므로 절대 건드리지 않는다.**
`rag_service.py`와 `test_document_reorder.py`는 목록에 없어야 한다.

- [ ] **Step 8: 기준선 테스트**

```bash
cd backend && app/venv/bin/python -m pytest tests/ -q 2>&1 | tail -3
```

기대: **`149 passed`** (156에서 삭제한 프로토타입 테스트 7개를 뺀 값)

커밋하지 않는다 — 되돌리기만 한 상태다.

---

## Task 2: `LongContextReorder` 계약 테스트

우리는 서드파티의 **정확한 순서 동작**에 의존한다. 이 테스트가 없으면 `langchain-community` 버전을 올렸을 때 배치가 바뀌어도 조용히 넘어간다. 구현보다 먼저 라이브러리가 실제로 무엇을 하는지 못박는다.

**Files:**
- Create: `backend/tests/test_long_context_reorder.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_long_context_reorder.py`:

```python
"""langchain_community의 LongContextReorder 동작을 계약으로 고정한다.

재배치 알고리즘을 직접 소유하지 않고 라이브러리에 의존하기로 했으므로
(docs/superpowers/specs/2026-09-02-search-result-reordering-design.md 2절),
버전을 올렸을 때 배치가 달라지면 이 테스트가 깨져서 알려줘야 한다.
"""

import pytest
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder


def docs_numbered(n):
    """관련성 내림차순 문서 n개. '1'이 가장 관련성 높다."""
    return [Document(page_content=str(i + 1), metadata={}) for i in range(n)]


def reorder(docs):
    return [d.page_content for d in LongContextReorder().transform_documents(docs)]


@pytest.mark.parametrize(
    "n, expected",
    [
        # 홀수: 1등이 맨 앞
        (5, ["1", "3", "5", "4", "2"]),
        (7, ["1", "3", "5", "7", "6", "4", "2"]),
        # 짝수: 방향이 뒤집혀 1등이 맨 뒤 (설계 문서 3.6절)
        (4, ["2", "4", "3", "1"]),
        (10, ["2", "4", "6", "8", "10", "9", "7", "5", "3", "1"]),
    ],
)
def test_reordering_is_stable_across_library_versions(n, expected):
    assert reorder(docs_numbered(n)) == expected


def test_most_relevant_documents_end_up_at_the_edges():
    """홀짝과 무관하게 성립해야 하는 본질적 성질.

    가장자리에서 안쪽으로 짝지어 보면 관련성 순서가 유지된다.
    예: 10개면 양 끝이 {1,2}, 그 안쪽이 {3,4}, ... 가운데가 {9,10}.
    """
    for n in (4, 5, 7, 10):
        result = [int(x) for x in reorder(docs_numbered(n))]
        for i in range(n // 2):
            pair = {result[i], result[n - 1 - i]}
            expected_pair = {2 * i + 1, 2 * i + 2}
            assert pair == expected_pair, f"n={n}, 바깥에서 {i}번째 쌍이 {pair}"


def test_empty_and_single_document_pass_through():
    assert reorder([]) == []
    assert reorder(docs_numbered(1)) == ["1"]


def test_input_list_is_not_mutated():
    """_litm_reordering이 내부에서 reverse()를 호출하므로 확인해 둔다."""
    docs = docs_numbered(5)
    LongContextReorder().transform_documents(docs)

    assert [d.page_content for d in docs] == ["1", "2", "3", "4", "5"]
```

- [ ] **Step 2: 테스트 실행**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_long_context_reorder.py -v 2>&1 | tail -15
```

기대: **7 passed** (parametrize 4개 + 나머지 3개). 이 테스트는 라이브러리의 현재 동작을 기록하는 것이라 처음부터 통과하는 것이 정상이다. 하나라도 실패하면 설치된 `langchain-community` 버전이 0.0.38이 아니므로 **진행을 멈추고 보고한다.**

```bash
app/venv/bin/python -c "import langchain_community; print(langchain_community.__version__)"
```

기대: `0.0.38`

- [ ] **Step 3: 커밋**

```bash
git add backend/tests/test_long_context_reorder.py
git commit -m "test: pin LongContextReorder's ordering as a library contract

재배치 알고리즘을 직접 구현하지 않고 langchain-community에 의존하기로 했으므로,
버전 업그레이드로 배치가 바뀌면 조용히 넘어가지 않도록 현재 동작을 고정한다.
입력 개수가 짝수면 방향이 뒤집힌다는 점도 함께 기록한다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: `retrieval_reorder` 설정 추가

**Files:**
- Modify: `backend/app/config.py` (`retrieval_k` 바로 아래)
- Test: `backend/tests/test_local_rag_defaults.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_local_rag_defaults.py` 맨 끝에 추가:

```python
def test_retrieval_reorder_is_enabled_by_default():
    """Lost in the Middle 완화는 기본으로 켜져 있고, 끌 수 있어야 한다."""
    settings = Settings()

    assert settings.retrieval_reorder is True
```

- [ ] **Step 2: 실패 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_local_rag_defaults.py::test_retrieval_reorder_is_enabled_by_default -v 2>&1 | tail -5
```

기대: FAIL — `AttributeError: 'Settings' object has no attribute 'retrieval_reorder'`

- [ ] **Step 3: 설정 추가**

`backend/app/config.py`에서 `retrieval_k: int = 10  # Number of chunks to retrieve` 줄 **바로 아래**에 추가:

```python
    # 검색 결과를 관련성 순서 그대로 넣지 않고, 관련성 높은 청크를 컨텍스트
    # 양 끝으로 보낸다(Lost in the Middle 완화). 끄면 관련성 내림차순 그대로.
    retrieval_reorder: bool = True
```

- [ ] **Step 4: 통과 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_local_rag_defaults.py -v 2>&1 | tail -8
```

기대: **4 passed**

- [ ] **Step 5: 커밋**

```bash
git add backend/app/config.py backend/tests/test_local_rag_defaults.py
git commit -m "feat: add RETRIEVAL_REORDER setting, on by default

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: `ScoringRetriever`에 재배치 적용

이 Task는 **기존 테스트 하나를 깨뜨린다.** `test_relevance_score_is_stashed_in_document_metadata`가 문서 2개로 `["a", "b"]`를 단언하는데, n=2에서 `LongContextReorder`는 `["b", "a"]`를 돌려준다. 그 테스트의 관심사는 순서가 아니라 **점수 부착**이므로 `reorder=False`로 관심사를 격리한다.

**Files:**
- Modify: `backend/app/services/rag_service.py` (import, 모듈 상수, `ScoringRetriever`, `ask_question`)
- Test: `backend/tests/test_scoring_retriever.py`

- [ ] **Step 1: 기존 테스트의 관심사 격리**

`backend/tests/test_scoring_retriever.py`의 `test_relevance_score_is_stashed_in_document_metadata`에서 리트리버 생성 줄을 다음으로 **교체**한다.

교체 전:

```python
    retriever = ScoringRetriever(vector_store=store, k=4)
```

교체 후:

```python
    # 이 테스트의 관심사는 점수 부착이지 순서가 아니다. 재배치를 끄고 격리한다.
    retriever = ScoringRetriever(vector_store=store, k=4, reorder=False)
```

- [ ] **Step 2: 실패하는 테스트 작성**

`backend/tests/test_scoring_retriever.py` 맨 끝에 추가:

```python
def test_reordering_moves_the_best_chunks_to_the_edges():
    """기본값(reorder=True)에서 관련성 높은 청크가 컨텍스트 양 끝으로 간다.

    기대값은 retrieval_k와 같은 10개 기준이며, LongContextReorder의 실제
    동작이다(tests/test_long_context_reorder.py가 이 계약을 고정한다).
    """
    store = FakeVectorStore([
        (Document(page_content=str(i + 1), metadata={}), 1.0 - i * 0.01)
        for i in range(10)
    ])
    retriever = ScoringRetriever(vector_store=store, k=10)

    docs = retriever.get_relevant_documents("질문")

    assert [d.page_content for d in docs] == [
        "2", "4", "6", "8", "10", "9", "7", "5", "3", "1"
    ]


def test_scores_survive_the_reordering():
    """재배치 후에도 점수가 각 문서를 따라다녀야 출처 정렬이 가능하다."""
    store = FakeVectorStore([
        (Document(page_content="a", metadata={}), 0.91),
        (Document(page_content="b", metadata={}), 0.42),
        (Document(page_content="c", metadata={}), 0.30),
    ])
    retriever = ScoringRetriever(vector_store=store, k=3)

    docs = retriever.get_relevant_documents("질문")
    by_content = {d.page_content: d.metadata["similarity_score"] for d in docs}

    assert by_content == {"a": 0.91, "b": 0.42, "c": 0.30}


def test_reorder_can_be_turned_off():
    store = FakeVectorStore([
        (Document(page_content="a", metadata={}), 0.91),
        (Document(page_content="b", metadata={}), 0.42),
    ])
    retriever = ScoringRetriever(vector_store=store, k=2, reorder=False)

    docs = retriever.get_relevant_documents("질문")

    assert [d.page_content for d in docs] == ["a", "b"]


def test_empty_search_result_does_not_blow_up():
    retriever = ScoringRetriever(vector_store=FakeVectorStore([]), k=10)

    assert retriever.get_relevant_documents("질문") == []
```

- [ ] **Step 3: 실패 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_scoring_retriever.py -v 2>&1 | tail -15
```

기대: `test_reordering_moves_the_best_chunks_to_the_edges`가 FAIL — 재배치가 없어 `["1","2",...,"10"]`이 나온다. `test_reorder_can_be_turned_off`는 `ScoringRetriever`에 `reorder` 필드가 없어 pydantic이 거부하므로 FAIL.

- [ ] **Step 4: import와 모듈 상수 추가**

`backend/app/services/rag_service.py`의 import 블록에서 `from langchain_community.chat_models import ChatOllama` 줄 **바로 아래**에 추가:

```python
from langchain_community.document_transformers import LongContextReorder
```

그리고 `settings = get_settings()` 줄 **바로 아래**에 추가:

```python

# LLM은 긴 컨텍스트의 가운데를 잘 놓친다(Liu et al. 2023, "Lost in the Middle").
# 관련성 높은 청크를 양 끝으로 보내 이 취약 구간을 피한다. 상태가 없는 객체라
# 요청마다 새로 만들 이유가 없다.
_LONG_CONTEXT_REORDER = LongContextReorder()
```

- [ ] **Step 5: `ScoringRetriever` 수정**

docstring 끝(`나온 점수를 그대로 출처 응답까지 이어지게 한다.` 다음 줄)에 문단을 추가하고, `reorder` 필드와 재배치 로직을 넣는다.

교체 전:

```python
    doc.metadata['similarity_score']에 채워, 답변 생성과 동일한 검색 결과에서
    나온 점수를 그대로 출처 응답까지 이어지게 한다.
    """

    vector_store: Any
    k: int
    search_filter: Optional[dict] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k, filter=self.search_filter
        )
        for doc, score in results:
            doc.metadata["similarity_score"] = score
        return [doc for doc, _ in results]
```

교체 후:

```python
    doc.metadata['similarity_score']에 채워, 답변 생성과 동일한 검색 결과에서
    나온 점수를 그대로 출처 응답까지 이어지게 한다.

    reorder=True면 관련성 높은 청크를 컨텍스트 양 끝으로 재배치한다. 점수는
    메타데이터에 남으므로, 사용자에게 보여줄 출처 목록은 _format_sources가
    다시 점수 순으로 되돌린다.
    """

    vector_store: Any
    k: int
    search_filter: Optional[dict] = None
    reorder: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k, filter=self.search_filter
        )
        # 점수는 반드시 재배치 "전"에 심는다. 재배치하면 문서 순서가 바뀌면서
        # results의 (doc, score) 짝을 더는 위치로 복원할 수 없다.
        for doc, score in results:
            doc.metadata["similarity_score"] = score

        docs = [doc for doc, _ in results]
        if not self.reorder:
            return docs
        # transform_documents는 Sequence를 돌려주므로 List로 맞춘다.
        return list(_LONG_CONTEXT_REORDER.transform_documents(docs))
```

- [ ] **Step 6: 설정 주입 테스트 작성**

`reorder` 필드를 만들어도 `RAGService.ask_question`이 설정을 넘기지 않으면
`RETRIEVAL_REORDER`는 아무 효과가 없다. 배선을 테스트로 고정한다.

`backend/tests/test_scoring_retriever.py` 맨 끝에 추가:

```python
def test_ask_question_injects_the_reorder_setting_into_the_retriever(monkeypatch):
    """RETRIEVAL_REORDER가 실제로 리트리버까지 전달되는지 확인한다.

    이 배선이 빠지면 설정을 꺼도 재배치가 계속 동작한다.
    """
    import asyncio
    from types import SimpleNamespace

    import app.services.rag_service as rag_module
    from app.services.rag_service import RAGService

    captured = {}

    class FakeChain:
        def __call__(self, inputs):
            return {"answer": "ok", "source_documents": []}

    def fake_from_llm(llm, retriever, **kwargs):
        captured["retriever"] = retriever
        return FakeChain()

    monkeypatch.setattr(
        rag_module.ConversationalRetrievalChain, "from_llm", staticmethod(fake_from_llm)
    )
    # 실제 Settings 대신 필요한 두 값만 가진 대역으로 갈아끼운다.
    monkeypatch.setattr(
        rag_module, "settings", SimpleNamespace(retrieval_k=10, retrieval_reorder=False)
    )

    # __init__은 임베딩 모델과 Ollama 연결을 요구하므로 우회한다.
    service = RAGService.__new__(RAGService)
    service.vector_store = FakeVectorStore([])
    service.llm = object()
    service.conversation_memories = {}

    asyncio.run(service.ask_question("질문"))

    assert captured["retriever"].reorder is False
    assert captured["retriever"].k == 10
```

- [ ] **Step 7: `ask_question`에 설정 주입**

`backend/app/services/rag_service.py`의 `ask_question` 안에서 리트리버를 만드는 부분을 수정한다.

교체 전:

```python
        retriever = ScoringRetriever(
            vector_store=self.vector_store,
            k=search_kwargs["k"],
            search_filter=search_kwargs.get("filter"),
        )
```

교체 후:

```python
        retriever = ScoringRetriever(
            vector_store=self.vector_store,
            k=search_kwargs["k"],
            search_filter=search_kwargs.get("filter"),
            reorder=settings.retrieval_reorder,
        )
```

- [ ] **Step 8: 통과 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_scoring_retriever.py -v 2>&1 | tail -14
```

기대: **8 passed** (기존 3 + 신규 5)

- [ ] **Step 9: 줄바꿈이 보존됐는지 확인**

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System
file backend/app/services/rag_service.py
git diff --numstat backend/app/services/rag_service.py
```

기대: `with CRLF line terminators`가 유지되고, 변경 줄 수는 **20줄 미만**이다. 200줄대가 나오면 파일 전체가 LF로 바뀐 것이므로 `git checkout -- backend/app/services/rag_service.py`로 되돌리고 Edit 도구로 다시 수정한다.

- [ ] **Step 10: 커밋**

```bash
git add backend/app/services/rag_service.py backend/tests/test_scoring_retriever.py
git commit -m "feat: reorder retrieved chunks to keep the best ones off-center

LLM은 긴 컨텍스트의 가운데를 잘 놓치므로(Liu et al. 2023) 관련성 높은 청크를
양 끝으로 보낸다. 점수는 재배치 전에 메타데이터로 심어 출처 정렬에 쓸 수 있게
남긴다.

RETRIEVAL_REORDER 설정을 ask_question에서 리트리버로 주입해 실제로 끌 수 있게 했다.

기존 점수 부착 테스트는 문서 2개로 순서를 단언하고 있었는데, 재배치가 켜지면
n=2에서 순서가 뒤집힌다. 그 테스트의 관심사는 순서가 아니므로 reorder=False로
격리했다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: 출처 목록을 점수순으로 되돌리기

`ConversationalRetrievalChain`은 리트리버가 돌려준 리스트를 컨텍스트와 `source_documents` 양쪽에 그대로 쓴다. 프론트엔드는 배열 순서대로 렌더링하므로, 조치하지 않으면 사용자가 `Relevance 62%`가 `91%`보다 위에 뜨는 화면을 보게 된다.

**Files:**
- Modify: `backend/app/services/rag_service.py` (`_format_sources`)
- Test: `backend/tests/test_source_formatting.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_source_formatting.py` 맨 끝에 추가:

```python
def test_sources_are_sorted_by_score_even_when_input_was_reordered():
    """리트리버가 재배치한 순서로 들어와도 출처는 관련성 순으로 보인다."""
    docs = [
        Document(page_content="중간", metadata={
            "filename": "a.pdf", "document_id": "doc_1",
            "chunk_index": 1, "similarity_score": 0.62,
        }),
        Document(page_content="최고", metadata={
            "filename": "a.pdf", "document_id": "doc_1",
            "chunk_index": 0, "similarity_score": 0.91,
        }),
        Document(page_content="최저", metadata={
            "filename": "a.pdf", "document_id": "doc_1",
            "chunk_index": 2, "similarity_score": 0.30,
        }),
    ]

    sources = _format(docs)

    assert [s.content for s in sources] == ["최고", "중간", "최저"]
    assert [s.similarity_score for s in sources] == [0.91, 0.62, 0.30]


def test_missing_similarity_score_sorts_last_without_crashing():
    """리트리버를 거치지 않고 들어온 문서가 정렬을 깨뜨리지 않아야 한다."""
    docs = [
        Document(page_content="점수 없음", metadata={
            "filename": "a.pdf", "document_id": "doc_1", "chunk_index": 0,
        }),
        Document(page_content="점수 있음", metadata={
            "filename": "a.pdf", "document_id": "doc_1",
            "chunk_index": 1, "similarity_score": 0.40,
        }),
    ]

    sources = _format(docs)

    assert [s.content for s in sources] == ["점수 있음", "점수 없음"]
```

- [ ] **Step 2: 실패 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_source_formatting.py -v 2>&1 | tail -10
```

기대: 두 테스트 모두 FAIL — 입력 순서가 그대로 나와 `["중간", "최고", "최저"]`가 된다.

- [ ] **Step 3: `_format_sources` 수정**

교체 전:

```python
    def _format_sources(self, source_docs: list) -> list[SourceDocument]:
        """Format source documents for response."""
        formatted_sources = []

        for doc in source_docs:
```

교체 후:

```python
    def _format_sources(self, source_docs: list) -> list[SourceDocument]:
        """Format source documents for response.

        리트리버가 컨텍스트 배치용으로 순서를 바꿨더라도(ScoringRetriever.reorder)
        사용자에게 보여줄 출처는 관련성이 높은 순서여야 하므로 점수로 되돌린다.
        """
        formatted_sources = []

        # 점수가 없는 문서(리트리버를 거치지 않은 경우)는 0.0으로 취급해 맨 뒤로.
        ordered_docs = sorted(
            source_docs,
            key=lambda doc: doc.metadata.get("similarity_score") or 0.0,
            reverse=True,
        )

        for doc in ordered_docs:
```

- [ ] **Step 4: 통과 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_source_formatting.py -v 2>&1 | tail -10
```

기대: 기존 테스트 포함 전부 통과

- [ ] **Step 5: 커밋**

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System
git diff --numstat backend/app/services/rag_service.py
git add backend/app/services/rag_service.py backend/tests/test_source_formatting.py
git commit -m "fix: keep the source list ordered by relevance after reordering

재배치는 LLM을 위한 내부 최적화이므로 사용자 UI로 새어나가면 안 된다.
프론트엔드가 배열 순서대로 렌더링하므로 백엔드에서 정렬한다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

`git diff --numstat`이 20줄 미만인지 먼저 확인한다.

---

## Task 6: 프롬프트에서 `(관련성 순)` 제거

`QA_PROMPT`가 컨텍스트를 `## 검색된 문서 (관련성 순)`이라고 소개한다. 재배치 후에는 사실이 아니며, LLM에게 "앞쪽이 더 관련성 높다"는 틀린 힌트를 준다.

**Files:**
- Modify: `backend/app/services/rag_service.py` (`QA_PROMPT`)
- Test: `backend/tests/test_local_rag_defaults.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_local_rag_defaults.py` 맨 끝에 추가:

```python
def test_qa_prompt_does_not_claim_the_context_is_relevance_ordered():
    """재배치 후에는 컨텍스트가 관련성 순이 아니므로 그렇게 말하면 안 된다."""
    from app.services.rag_service import QA_PROMPT

    assert "관련성 순" not in QA_PROMPT.template
```

- [ ] **Step 2: 실패 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_local_rag_defaults.py::test_qa_prompt_does_not_claim_the_context_is_relevance_ordered -v 2>&1 | tail -5
```

기대: FAIL — `assert '관련성 순' not in ...`

- [ ] **Step 3: 프롬프트 수정**

교체 전:

```
## 검색된 문서 (관련성 순)
```

교체 후:

```
## 검색된 문서
```

- [ ] **Step 4: 통과 확인**

```bash
cd backend && app/venv/bin/python -m pytest tests/test_local_rag_defaults.py -v 2>&1 | tail -8
```

기대: **5 passed**

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/rag_service.py backend/tests/test_local_rag_defaults.py
git commit -m "fix: stop telling the LLM the context is relevance-ordered

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: 설정 문서화

**Files:**
- Modify: `backend/.env.example` (`RETRIEVAL_K=10` 다음 줄)
- Modify: `README.md` (설정 표, `RETRIEVAL_K` 행 다음)

- [ ] **Step 1: `.env.example` 수정**

`RETRIEVAL_K=10` 줄 **바로 아래**에 추가:

```
RETRIEVAL_REORDER=true  # 관련성 높은 청크를 컨텍스트 앞/뒤 끝에 배치 (Lost in the Middle 완화)
```

- [ ] **Step 2: `README.md` 설정 표 수정**

표는 **영문**이다. `| `RETRIEVAL_K` | `10` | Chunks retrieved per question |` 줄 **바로 아래**에 추가:

```
| `RETRIEVAL_REORDER` | `true` | Put the most relevant chunks at both ends of the context |
```

- [ ] **Step 3: 확인**

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System
grep -n "RETRIEVAL_REORDER" backend/.env.example README.md
```

기대: 두 파일에서 각각 한 줄씩 나온다.

- [ ] **Step 4: 커밋**

```bash
git add backend/.env.example README.md
git commit -m "docs: document RETRIEVAL_REORDER

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

`README.md`에는 이 작업과 무관한 기존 변경분이 있을 수 있다. `git diff --cached README.md`로 표 한 줄만 스테이징됐는지 확인하고, 아니면 `git add -p README.md`로 해당 줄만 담는다.

---

## Task 8: 전체 검증

- [ ] **Step 1: 전체 스위트 실행**

```bash
cd backend && app/venv/bin/python -m pytest tests/ -q 2>&1 | tail -5
```

기대: **165 passed** (기준선 149 + 계약 7 + 리트리버 5 + 출처 2 + 설정 1 + 프롬프트 1)

계산이 어긋나면 실제 숫자를 보고한다. 중요한 것은 **failed가 0**이고 149보다 줄지 않는 것이다.

- [ ] **Step 2: 줄바꿈 최종 확인**

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System
file backend/app/services/rag_service.py
```

기대: `with CRLF line terminators`

- [ ] **Step 3: 이 작업의 커밋만 담겼는지 확인**

```bash
git log --oneline -6
git status --short backend/
```

기대: `chunking.py`, `document_processor.py`, `mineru_client.py`, `test_chunking.py`, `test_mineru_client.py`가 여전히 **커밋되지 않은 채** 남아 있다. 이들은 다른 작업의 변경분이다.

---

## Task 9: 수동 A/B 관찰

단위 테스트는 순서가 의도대로 바뀌는 것만 증명한다. 답변이 실제로 나아지는지는 별개다.

- [ ] **Step 1: 백엔드 기동 (재배치 켠 상태)**

```bash
cd backend && app/venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 2: 질문 3개를 던지고 답변 저장**

이미 인덱싱된 문서에 대해, **문서 뒷부분이나 중간에 근거가 있을 법한** 질문을 고른다. 근거가 1페이지에 있으면 재배치와 무관하게 잘 답하므로 차이가 안 보인다.

각 질문의 답변과 출처 목록을 스크래치 파일에 기록한다.

- [ ] **Step 3: 재배치를 끄고 재기동**

`backend/.env`에 다음을 추가(파일이 없으면 `.env.example`을 복사해서 만든다):

```
RETRIEVAL_REORDER=false
```

서버를 재시작한다. `get_settings()`가 `@lru_cache`이므로 **`--reload`만으로는 반영되지 않는다.** 프로세스를 완전히 종료하고 다시 띄운다.

- [ ] **Step 4: 같은 질문 3개를 다시 던지고 답변 저장**

- [ ] **Step 5: `.env` 원복**

```bash
grep -v "^RETRIEVAL_REORDER=" backend/.env > backend/.env.tmp && mv backend/.env.tmp backend/.env
```

- [ ] **Step 6: 관찰 기록**

`docs/troubleshooting/2026-09-02-search-result-reordering-ab.md`에 기록한다. 기존 트러블슈팅 문서 형식(요약 / 방법 / 결과 / 한계)을 따른다.

**개선이 관찰되지 않으면 그 사실을 그대로 쓴다.** 설계 문서 6절에 이미 "효과는 측정된 것이 아니라 사례로 관찰한 것"이라고 명시했다. 없는 효과를 지어내지 않는다.

- [ ] **Step 7: 커밋**

```bash
git add docs/troubleshooting/2026-09-02-search-result-reordering-ab.md
git commit -m "docs: record the reordering A/B observation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
