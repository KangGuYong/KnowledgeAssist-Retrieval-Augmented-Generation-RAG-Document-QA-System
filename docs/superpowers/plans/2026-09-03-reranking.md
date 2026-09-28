# 크로스인코더 리랭킹 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 벡터 검색으로 후보 30개를 뽑고 크로스인코더로 재채점해 상위 5개만 LLM에 넘긴다.

**Architecture:** `services/reranker.py`가 모델을 소유하고(`@lru_cache` 게터, 지연 로드), `ScoringRetriever._get_relevant_documents`가 *검색 → 벡터 점수 부착 → 리랭킹 → 재배치* 순서를 한 메서드에서 관장한다. 리랭커가 실패하면 벡터 순서로 폴백해 질문은 성공시킨다. `rerank_enabled=False`면 Chroma에 넘기는 `k`부터 `retrieval_k`가 되어 현재와 완전히 동일하게 동작한다.

**Tech Stack:** Python 3.12, sentence-transformers 3.3.1 (`CrossEncoder`, 이미 설치됨), langchain-core 1.6.1, pytest 7.4.4, React 18 / TypeScript 5.3

**설계 문서:** [2026-09-03-reranking-design.md](../specs/2026-09-03-reranking-design.md)

---

## 사전 준비

**Python은 반드시 프로젝트 venv로 실행한다.** 시스템 `python`에는 의존성이 없다.

```bash
cd /itos-llm/KnowledgeAssist-Retrieval-Augmented-Generation-RAG-Document-QA-System/backend
./app/venv/bin/python -m pytest -q
```

착수 전 기준선: **185 passed**. 이 숫자가 다르면 멈추고 원인을 확인한다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `backend/app/services/reranker.py` (신규) | 크로스인코더 모델 소유, (질문, 청크) 재채점 |
| `backend/tests/test_reranker.py` (신규) | 리랭커 모듈 단위 테스트 |
| `backend/app/config.py` | 설정 4개 추가 |
| `backend/app/services/rag_service.py` | `ScoringRetriever` 리랭킹 단계, `_format_sources` 정렬 기준 |
| `backend/app/api/models/responses.py` | `SourceDocument.rerank_score` |
| `backend/tests/test_scoring_retriever.py` | 리랭킹 동작 테스트, settings 대역 갱신 |
| `backend/tests/test_source_formatting.py` | 정렬 기준 테스트 |
| `backend/tests/test_rag_chain.py` | settings 대역 갱신 |
| `frontend/src/types/api.types.ts` | `rerank_score` 필드 |
| `frontend/src/components/SourceCitation.tsx` | 두 점수 표시 |
| `backend/.env.example`, `README.md`, `ARCHITECTURE.md` | 설정 문서화 |

---

## Task 1: 설정 4개 추가

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_local_rag_defaults.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_local_rag_defaults.py` 끝에 추가:

```python
def test_reranking_defaults():
    """리랭킹은 기본으로 켜져 있고, 후보는 최종보다 많아야 의미가 있다."""
    settings = Settings(_env_file=None)

    assert settings.rerank_enabled is True
    assert settings.rerank_model == "dragonkue/bge-reranker-v2-m3-ko"
    assert settings.rerank_device == "cuda"
    assert settings.rerank_candidate_k == 30
    assert settings.rerank_candidate_k > settings.retrieval_k
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_local_rag_defaults.py::test_reranking_defaults -v
```

기대: FAIL — `AttributeError: 'Settings' object has no attribute 'rerank_enabled'`

- [ ] **Step 3: 설정을 추가한다**

`backend/app/config.py`의 `retrieval_reorder` 줄 **바로 아래**에 삽입:

```python
    # 리랭킹 - 후보를 넓게 뽑아 크로스인코더로 재채점한 뒤 상위 retrieval_k개만 쓴다.
    # 끄면 후보 수가 retrieval_k로 줄어 현재와 완전히 같은 동작이 된다.
    rerank_enabled: bool = True
    rerank_model: str = "dragonkue/bge-reranker-v2-m3-ko"
    rerank_device: str = "cuda"  # "cpu", "cuda", "cuda:0", ...
    rerank_candidate_k: int = 30  # 리랭커에 넘길 후보 수
```

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_local_rag_defaults.py -v
```

기대: PASS (기존 5개 + 신규 1개 = 6 passed)

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/config.py backend/tests/test_local_rag_defaults.py
git commit -m "feat: add reranking settings"
```

---

## Task 2: 리랭커 모듈

**Files:**
- Create: `backend/app/services/reranker.py`
- Test: `backend/tests/test_reranker.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_reranker.py` 신규 생성:

```python
"""크로스인코더 리랭커. 실제 모델은 로드하지 않고 대역으로 검증한다.

이 모듈이 지켜야 할 것은 두 가지다. (질문, 청크 본문) 쌍을 만들어 모델에
넘기는 것, 그리고 모델을 첫 사용 시점까지 만들지 않는 것.
"""

from langchain_core.documents import Document

from app.services.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """predict()에 들어온 쌍을 기록하고 정해진 점수를 돌려준다."""

    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores


def _reranker_with(model):
    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    reranker._model = model
    return reranker


def test_score_pairs_the_query_with_each_chunk():
    model = FakeCrossEncoder([0.9, 0.1])
    reranker = _reranker_with(model)
    docs = [Document(page_content="첫 청크"), Document(page_content="둘째 청크")]

    scores = reranker.score("생활SOC란?", docs)

    assert model.pairs == [("생활SOC란?", "첫 청크"), ("생활SOC란?", "둘째 청크")]
    assert scores == [0.9, 0.1]


def test_empty_document_list_does_not_touch_the_model():
    model = FakeCrossEncoder([])
    reranker = _reranker_with(model)

    assert reranker.score("질문", []) == []
    assert model.pairs is None


def test_model_is_not_built_until_first_use():
    """2.27GB 모델을 임포트만으로 올리면 안 된다."""
    reranker = CrossEncoderReranker(model_name="fake", device="cpu")

    assert reranker._model is None
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_reranker.py -v
```

기대: FAIL — `ModuleNotFoundError: No module named 'app.services.reranker'`

- [ ] **Step 3: 모듈을 만든다**

`backend/app/services/reranker.py` 신규 생성:

```python
"""크로스인코더 리랭커.

벡터 검색은 질문과 청크를 각각 따로 벡터화해 거리를 재므로 둘을 함께 읽지
않는다. 크로스인코더는 (질문, 청크) 쌍을 한 번에 입력받아 관련성을 직접
채점하므로, 벡터 검색이 뒤로 밀어낸 청크를 끌어올릴 수 있다.
"""

from functools import lru_cache
from typing import List, Optional
import logging
import threading

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """(질문, 청크) 쌍을 0~1로 채점한다.

    sentence-transformers의 CrossEncoder는 config.num_labels가 1이면
    predict()에 nn.Sigmoid()를 자동 적용한다(CrossEncoder.py:141). 기본 모델인
    dragonkue/bge-reranker-v2-m3-ko는 id2label이 하나라 여기 해당하므로 점수가
    0~1로 나오고, 화면의 퍼센트 표시에 그대로 쓸 수 있다. num_labels가 1이 아닌
    모델로 바꾸면 점수 범위가 달라져 그 표시가 어긋난다.
    """

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None
    ):
        self.model_name = model_name or settings.rerank_model
        self.device = device or settings.rerank_device
        self._model = None
        self._lock = threading.Lock()

    def _build_model(self):
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker: %s (device=%s)", self.model_name, self.device
        )
        return CrossEncoder(self.model_name, device=self.device)

    @property
    def model(self):
        """모델은 첫 채점 때 만든다. 2.27GB라 임포트 시점에 올릴 수 없다."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._build_model()
        return self._model

    def score(self, query: str, docs: List[Document]) -> List[float]:
        """docs와 같은 길이의 점수 목록을 돌려준다."""
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        return [float(score) for score in self.model.predict(pairs)]


@lru_cache()
def get_reranker() -> CrossEncoderReranker:
    """Cached reranker so the model is only loaded once per worker."""
    return CrossEncoderReranker()
```

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_reranker.py -v
```

기대: 3 passed

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/services/reranker.py backend/tests/test_reranker.py
git commit -m "feat: add the cross-encoder reranker module"
```

---

## Task 3: `ScoringRetriever`에 리랭킹 단계

**Files:**
- Modify: `backend/app/services/rag_service.py`
- Test: `backend/tests/test_scoring_retriever.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_scoring_retriever.py` 끝에 추가:

```python
class FakeReranker:
    """호출을 기록하고 정해진 점수를 돌려준다. 예외를 던지게도 만들 수 있다."""

    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error
        self.calls = []

    def score(self, query, docs):
        self.calls.append((query, [d.page_content for d in docs]))
        if self.error:
            raise self.error
        return self.scores


def _docs_with_scores(pairs):
    return [
        (Document(page_content=text, metadata={}), score) for text, score in pairs
    ]


def test_reranking_picks_the_top_k_by_rerank_score_not_vector_score():
    """벡터 순서와 리랭크 순서를 일부러 어긋나게 세운다.

    두 기준이 같은 답을 내면 무엇을 검증했는지 알 수 없다.
    """
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7)]))
    reranker = FakeReranker(scores=[0.1, 0.99, 0.5])
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=True, candidate_k=3,
        reranker=reranker, reorder=False,
    )

    docs = retriever.invoke("질문")

    assert [d.page_content for d in docs] == ["b", "c"]
    assert docs[0].metadata["rerank_score"] == 0.99
    # 벡터 점수도 그대로 남는다.
    assert docs[0].metadata["similarity_score"] == 0.8


def test_candidate_k_is_what_reaches_the_vector_store_when_reranking():
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7)]))
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=True, candidate_k=3,
        reranker=FakeReranker(scores=[0.1, 0.2, 0.3]), reorder=False,
    )

    retriever.invoke("질문")

    assert store.calls[0]["k"] == 3


def test_k_is_what_reaches_the_vector_store_when_not_reranking():
    """끄면 후보를 넓게 뽑지 않는다. 검색 비용까지 현재와 같아야 한다."""
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8)]))
    reranker = FakeReranker(scores=[0.1, 0.2])
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=False, candidate_k=30,
        reranker=reranker, reorder=False,
    )

    retriever.invoke("질문")

    assert store.calls[0]["k"] == 2
    assert reranker.calls == []


def test_a_failing_reranker_falls_back_to_vector_order():
    """리랭킹은 품질 개선이지 정확성 요건이 아니다. 질문이 실패하면 안 된다."""
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7)]))
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=True, candidate_k=3,
        reranker=FakeReranker(error=RuntimeError("모델 로드 실패")), reorder=False,
    )

    docs = retriever.invoke("질문")

    assert [d.page_content for d in docs] == ["a", "b"]
    assert "rerank_score" not in docs[0].metadata


def test_a_score_count_mismatch_falls_back_too():
    """점수 개수가 문서 수와 다르면 zip이 조용히 잘라내므로 폴백한다."""
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7)]))
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=True, candidate_k=3,
        reranker=FakeReranker(scores=[0.1]), reorder=False,
    )

    docs = retriever.invoke("질문")

    assert [d.page_content for d in docs] == ["a", "b"]
    assert "rerank_score" not in docs[0].metadata


def test_rerank_scores_survive_the_reordering():
    """재배치는 순서만 바꾼다. 두 점수 모두 남아야 _format_sources가 되돌린다."""
    store = FakeVectorStore(
        _docs_with_scores([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)])
    )
    retriever = ScoringRetriever(
        vector_store=store, k=4, rerank=True, candidate_k=4,
        reranker=FakeReranker(scores=[0.1, 0.4, 0.3, 0.2]), reorder=True,
    )

    docs = retriever.invoke("질문")

    assert all("rerank_score" in d.metadata for d in docs)
    assert all("similarity_score" in d.metadata for d in docs)
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_scoring_retriever.py -v
```

기대: 신규 6개 FAIL — `ValidationError` (`rerank`, `candidate_k`, `reranker`는 아직 필드가 아님)

- [ ] **Step 3: 리랭킹 단계를 구현한다**

`backend/app/services/rag_service.py`의 임포트 블록에서 `from langchain_ollama import ChatOllama` **바로 아래**에 추가:

```python
from app.services.reranker import get_reranker
```

`ScoringRetriever`의 필드 선언을 교체한다. 기존:

```python
    vector_store: Any
    k: int
    search_filter: Optional[dict] = None
    reorder: bool = True
```

교체 후:

```python
    vector_store: Any
    k: int
    search_filter: Optional[dict] = None
    reorder: bool = True
    rerank: bool = False
    candidate_k: int = 0
    # 기본값은 공유 리랭커(get_reranker). 테스트에서 주입할 수 있게 열어 둔다.
    reranker: Any = None
```

`_get_relevant_documents`를 교체한다. 기존:

```python
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

교체 후:

```python
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 리랭킹을 끄면 후보를 넓게 뽑지 않는다. 검색 비용까지 현재와 같아진다.
        # candidate_k가 0이면(설정을 빠뜨린 호출) Chroma가 TypeError를 던지고,
        # 이 호출은 _rerank의 try 바깥이라 질문 전체가 실패한다. k로 물러난다.
        fetch_k = (self.candidate_k or self.k) if self.rerank else self.k
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter=self.search_filter
        )
        # 점수는 반드시 재배치 "전"에 심는다. 재배치하면 문서 순서가 바뀌면서
        # results의 (doc, score) 짝을 더는 위치로 복원할 수 없다.
        for doc, score in results:
            doc.metadata["similarity_score"] = score

        docs = [doc for doc, _ in results]
        if self.rerank:
            docs = self._rerank(query, docs)

        if not self.reorder:
            return docs
        # transform_documents는 Sequence를 돌려주므로 List로 맞춘다.
        return list(_LONG_CONTEXT_REORDER.transform_documents(docs))

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """크로스인코더로 재채점해 상위 k개만 남긴다.

        리랭킹은 품질 개선이지 정확성 요건이 아니다. 모델 로드 실패든 추론
        실패든, 실패하면 벡터 순서 상위 k개로 물러나고 질문은 성공시킨다.
        """
        try:
            reranker = self.reranker or get_reranker()
            scores = reranker.score(query, docs)
            if len(scores) != len(docs):
                # zip이 조용히 잘라내는 대신 여기서 잡는다.
                raise ValueError(
                    f"reranker returned {len(scores)} scores for {len(docs)} documents"
                )
        except Exception as exc:
            logger.warning(
                "Reranking failed, using vector order: %s", exc, exc_info=True
            )
            return docs[: self.k]

        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = score
        ranked = sorted(
            docs, key=lambda doc: doc.metadata["rerank_score"], reverse=True
        )
        return ranked[: self.k]
```

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_scoring_retriever.py -v
```

또한 `candidate_k`를 빠뜨린 호출이 Chroma에 `k=0`을 넘기지 않는지 고정한다:

```python
def test_rerank_without_candidate_k_falls_back_to_k():
    """candidate_k를 빠뜨린 호출이 Chroma에 k=0을 넘겨 질문을 죽이면 안 된다."""
    store = FakeVectorStore(_docs_with_scores([("a", 0.9), ("b", 0.8)]))
    retriever = ScoringRetriever(
        vector_store=store, k=2, rerank=True,
        reranker=FakeReranker(scores=[0.1, 0.2]), reorder=False,
    )

    retriever.invoke("질문")

    assert store.calls[0]["k"] == 2
```

기대: 기존 10개 + 신규 7개 = 17 passed

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/services/rag_service.py backend/tests/test_scoring_retriever.py
git commit -m "feat: rerank retrieved candidates down to retrieval_k"
```

---

## Task 4: `RAGService`가 설정을 리트리버로 전달

**Files:**
- Modify: `backend/app/services/rag_service.py`
- Test: `backend/tests/test_scoring_retriever.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_scoring_retriever.py`의 `test_ask_question_injects_the_reorder_setting_into_the_retriever` **바로 아래**에 추가:

```python
@pytest.mark.parametrize("configured", [True, False])
def test_ask_question_injects_the_rerank_settings_into_the_retriever(
    monkeypatch, configured
):
    """RERANK_ENABLED와 후보 수가 실제로 리트리버까지 전달되는지 확인한다.

    한쪽 방향만 검사하면 rerank를 상수로 박아넣은 코드를 잡지 못한다.
    """
    import asyncio
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    import app.services.rag_service as rag_module
    from app.services.rag_service import RAGService

    captured = {}
    real_retriever_cls = rag_module.ScoringRetriever

    def spy(**kwargs):
        captured["retriever"] = real_retriever_cls(**kwargs)
        return captured["retriever"]

    monkeypatch.setattr(rag_module, "ScoringRetriever", spy)
    monkeypatch.setattr(
        rag_module, "settings",
        SimpleNamespace(
            retrieval_k=5,
            retrieval_reorder=True,
            ocr_block_prefix="[이미지 텍스트]",
            llm_max_attempts=1,
            rerank_enabled=configured,
            rerank_candidate_k=30,
        ),
    )

    service = RAGService.__new__(RAGService)
    service.vector_store = FakeVectorStore([])
    service.llm = RunnableLambda(lambda _: AIMessage(content="ok"))
    service.conversation_histories = {}

    asyncio.run(service.ask_question("질문"))

    assert captured["retriever"].rerank is configured
    assert captured["retriever"].candidate_k == 30
    assert captured["retriever"].k == 5
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_scoring_retriever.py -k rerank_settings -v
```

기대: FAIL — `assert False is True` (`rerank`가 기본값 `False`로 남아 있음)

- [ ] **Step 3: 전달을 구현한다**

`backend/app/services/rag_service.py`의 `ask_question` 안, 리트리버 생성 부분을 교체한다. 기존:

```python
        retriever = ScoringRetriever(
            vector_store=self.vector_store,
            k=settings.retrieval_k,
            search_filter=search_filter,
            reorder=settings.retrieval_reorder,
        )
```

교체 후:

```python
        retriever = ScoringRetriever(
            vector_store=self.vector_store,
            k=settings.retrieval_k,
            search_filter=search_filter,
            reorder=settings.retrieval_reorder,
            rerank=settings.rerank_enabled,
            candidate_k=settings.rerank_candidate_k,
        )
```

- [ ] **Step 4: 통과를 확인한다**

기존 테스트의 settings 대역에도 새 필드가 필요하다. `test_scoring_retriever.py`의
`test_ask_question_injects_the_reorder_setting_into_the_retriever` 안 `SimpleNamespace(...)`에
두 줄을 더한다:

```python
            rerank_enabled=False,
            rerank_candidate_k=30,
```

`backend/tests/test_rag_chain.py`의 `build_service` 안 `SimpleNamespace(...)`에도 같은 두 줄을 더한다:

```python
            rerank_enabled=False,
            rerank_candidate_k=30,
```

그리고 실행한다:

```bash
./app/venv/bin/python -m pytest -q
```

기대: **198 passed** (기준선 185 + Task1 1개 + Task2 3개 + Task3 7개 + Task4 2개)

이 시점에서 설계 문서 §7.2가 요구하는 것이 확인된다: 새 테스트의 settings 대역이
전부 `rerank_enabled=False`이므로, **기존 테스트는 한 줄도 수정하지 않고 통과해야
한다.** 기존 테스트 본문을 손대야 한다면 끈 상태의 동작이 바뀐 것이므로 멈추고
이유를 확인한다(settings 대역에 필드를 더하는 것은 본문 수정이 아니다).

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/services/rag_service.py backend/tests/test_scoring_retriever.py backend/tests/test_rag_chain.py
git commit -m "feat: pass the rerank settings through to the retriever"
```

---

## Task 5: 출처 정렬 기준

**Files:**
- Modify: `backend/app/services/rag_service.py`
- Test: `backend/tests/test_source_formatting.py`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`backend/tests/test_source_formatting.py` 끝에 추가:

이 파일에는 이미 `_format(docs)` 헬퍼가 있다(`RAGService.__new__`로 우회해
`_format_sources`만 부른다). 그것을 그대로 쓴다.

```python
def test_sources_are_sorted_by_rerank_score_when_present():
    """리랭커가 선별했으면 출처 순서도 리랭크 점수를 따라야 한다.

    벡터 점수로 정렬하면 화면의 순서와 선별 근거가 어긋난다. 두 점수를 일부러
    반대로 세워 어느 기준이 쓰였는지 구분한다.
    """
    sources = _format([
        Document(page_content="a", metadata={"similarity_score": 0.9, "rerank_score": 0.1}),
        Document(page_content="b", metadata={"similarity_score": 0.5, "rerank_score": 0.99}),
    ])

    assert [s.content for s in sources] == ["b", "a"]
    assert sources[0].rerank_score == 0.99


def test_sources_fall_back_to_similarity_score_without_reranking():
    """리랭커를 껐거나 폴백했으면 기존 기준 그대로다."""
    sources = _format([
        Document(page_content="a", metadata={"similarity_score": 0.5}),
        Document(page_content="b", metadata={"similarity_score": 0.9}),
    ])

    assert [s.content for s in sources] == ["b", "a"]
    assert sources[0].rerank_score is None
```

- [ ] **Step 2: 실패를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_source_formatting.py -k rerank -v
```

기대: FAIL — 첫 테스트가 `["a", "b"]`를 얻는다(벡터 점수 순), 그리고 `SourceDocument`에 `rerank_score` 필드가 없다

- [ ] **Step 3: 정렬 기준과 응답 필드를 구현한다**

`backend/app/api/models/responses.py`의 `SourceDocument`에서 `similarity_score` 줄 **바로 아래**에 추가:

```python
    rerank_score: Optional[float] = Field(None, description="Cross-encoder relevance score")
```

`backend/app/services/rag_service.py`의 `_format_sources`에서 정렬 부분을 교체한다. 기존:

```python
        ordered_docs = sorted(
            source_docs,
            key=lambda doc: doc.metadata.get("similarity_score") or float("-inf"),
            reverse=True,
        )
```

교체 후:

```python
        # 리랭킹을 거친 응답은 리랭크 점수가 기준이다. 한 응답의 문서는 전부
        # 리랭킹을 거쳤거나 전부 거치지 않았거나 둘 중 하나이므로(폴백도 요청
        # 단위다) 두 점수 체계가 한 정렬에 섞이지 않는다.
        score_key = (
            "rerank_score"
            if any("rerank_score" in doc.metadata for doc in source_docs)
            else "similarity_score"
        )
        ordered_docs = sorted(
            source_docs,
            key=lambda doc: doc.metadata.get(score_key) or float("-inf"),
            reverse=True,
        )
```

같은 메서드의 `SourceDocument(...)` 생성에서 `similarity_score=...` 줄 **바로 아래**에 추가:

```python
                rerank_score=doc.metadata.get("rerank_score"),
```

- [ ] **Step 4: 통과를 확인한다**

```bash
./app/venv/bin/python -m pytest tests/test_source_formatting.py -v
```

기대: 기존 8개 + 신규 2개 = 10 passed

- [ ] **Step 5: 커밋한다**

```bash
git add backend/app/services/rag_service.py backend/app/api/models/responses.py backend/tests/test_source_formatting.py
git commit -m "feat: order sources by rerank score and expose it in the response"
```

---

## Task 6: 화면에 두 점수 표시

**Files:**
- Modify: `frontend/src/types/api.types.ts`
- Modify: `frontend/src/components/SourceCitation.tsx`

- [ ] **Step 1: 타입을 추가한다**

`frontend/src/types/api.types.ts`의 `SourceDocument`에서 `similarity_score: number | null;` **바로 아래**에 추가:

```typescript
  rerank_score: number | null;
```

- [ ] **Step 2: 표시를 바꾼다**

`frontend/src/components/SourceCitation.tsx`에서 다음 블록을 찾는다:

```tsx
      {source.similarity_score != null && (
        <div className="source-score">
          Relevance: {(source.similarity_score * 100).toFixed(1)}%
        </div>
      )}
```

교체한다:

```tsx
      {source.similarity_score != null && (
        <div className="source-score">
          Relevance: {(source.similarity_score * 100).toFixed(1)}%
          {source.rerank_score != null && (
            <> · Rerank: {(source.rerank_score * 100).toFixed(1)}%</>
          )}
        </div>
      )}
```

리랭커를 껐거나 폴백했을 때는 `rerank_score`가 `null`이라 지금과 똑같이 `Relevance`만 나온다.

- [ ] **Step 3: 타입 검사를 통과하는지 확인한다**

```bash
cd ../frontend && npx tsc --noEmit
```

기대: 출력 없음(통과)

- [ ] **Step 4: 커밋한다**

```bash
cd .. && git add frontend/src/types/api.types.ts frontend/src/components/SourceCitation.tsx
git commit -m "feat: show the rerank score beside the vector relevance"
```

---

## Task 7: 문서화

**Files:**
- Modify: `backend/.env.example`
- Modify: `README.md:197-198` 주변 설정표
- Modify: `ARCHITECTURE.md` 설정표 및 질의 흐름

- [ ] **Step 1: `.env.example`에 추가한다**

`backend/.env.example`의 `RETRIEVAL_REORDER=true` 줄 **바로 아래**에 삽입:

```
RERANK_ENABLED=true                      # 후보를 넓게 뽑아 크로스인코더로 재채점 후 상위 RETRIEVAL_K개만 사용
RERANK_MODEL="dragonkue/bge-reranker-v2-m3-ko"  # 한국어 추가 학습된 다국어 리랭커 (2.27GB)
RERANK_DEVICE="cuda"                     # cpu, cuda, cuda:0 ...
RERANK_CANDIDATE_K=30                    # 리랭커에 넘길 후보 수. 끄면 RETRIEVAL_K만 뽑는다
```

- [ ] **Step 2: `README.md` 설정표에 추가한다**

`README.md`의 `| `RETRIEVAL_REORDER` | `true` | Put the most relevant chunks at both ends of the context |` 줄 **바로 아래**에 삽입:

```markdown
| `RERANK_ENABLED` | `true` | Rescore a wider candidate pool with a cross-encoder, keep `RETRIEVAL_K` |
| `RERANK_MODEL` | `dragonkue/bge-reranker-v2-m3-ko` | Korean-tuned multilingual reranker (2.27GB) |
| `RERANK_DEVICE` | `cuda` | `cpu`, `cuda`, `cuda:0`, … |
| `RERANK_CANDIDATE_K` | `30` | Candidates fetched before reranking |
```

- [ ] **Step 3: `ARCHITECTURE.md`를 갱신한다**

설정표에서 `| `retrieval_k` | `10` | 검색할 청크 수 |` 줄 **바로 아래**에 삽입:

```markdown
| `rerank_enabled` | `true` | 크로스인코더 재채점 사용 |
| `rerank_candidate_k` | `30` | 재채점 전에 뽑는 후보 수 |
```

그리고 질의 흐름 설명에서 `similarity search` 단계 서술에 리랭킹을 반영한다. 다음 문장을 찾는다:

```
검색과 답변 모두 재작성된 질문을 쓴다.
```

바로 앞에 한 문단을 삽입한다:

```
`rerank_enabled`가 켜져 있으면 벡터 검색이 `rerank_candidate_k`개를 뽑고,
크로스인코더가 (질문, 청크) 쌍을 재채점해 상위 `retrieval_k`개만 남긴다.
리랭커가 실패하면 벡터 순서로 물러나며 질문은 실패하지 않는다.
```

- [ ] **Step 4: `.env.example`이 여전히 파싱되는지 확인한다**

```bash
cd backend && ./app/venv/bin/python -c "
from app.config import Settings
s = Settings(_env_file='.env.example')
print('rerank_enabled     :', s.rerank_enabled)
print('rerank_model       :', s.rerank_model)
print('rerank_device      :', s.rerank_device)
print('rerank_candidate_k :', s.rerank_candidate_k)
"
```

기대:
```
rerank_enabled     : True
rerank_model       : dragonkue/bge-reranker-v2-m3-ko
rerank_device      : cuda
rerank_candidate_k : 30
```

- [ ] **Step 5: 커밋한다**

```bash
cd .. && git add backend/.env.example README.md ARCHITECTURE.md
git commit -m "docs: document the reranking settings"
```

---

## Task 8: 수동 A/B (머지 전 필수)

설계 문서 §7.3이 요구하는 관찰이다. **이 태스크 없이 머지하지 않는다.**

- [ ] **Step 1: 전체 테스트가 통과하는지 확인한다**

```bash
cd backend && ./app/venv/bin/python -m pytest -q
```

기대: 실패 0

- [ ] **Step 2: 리랭커가 실제로 로드되는지 확인한다**

모델 2.27GB를 처음 내려받으므로 시간이 걸린다.

```bash
./app/venv/bin/python -c "
import time
from langchain_core.documents import Document
from app.services.reranker import CrossEncoderReranker
r = CrossEncoderReranker()
docs = [Document(page_content='생활SOC 확충 계획의 예산은 3,200억 원이다'),
        Document(page_content='이 문서는 도시계획 절차를 설명한다')]
t = time.perf_counter(); scores = r.score('생활SOC 예산이 얼마인가?', docs)
print(f'로드+채점 {time.perf_counter()-t:.1f}s  점수={scores}')
"
```

기대: 첫 점수가 둘째보다 높고, **두 값 모두 0~1 범위**여야 한다. 범위를 벗어나면 모델의 `num_labels`가 1이 아니라는 뜻이므로 멈추고 보고한다(화면 퍼센트 표시가 깨진다).

- [ ] **Step 3: 서버를 띄우고 켠 상태로 질문한다**

```bash
./app/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

브라우저에서 같은 문서에 질문 3~4개를 던지고 **출처 목록과 답변, 응답 시간**을 기록한다.

- [ ] **Step 4: 끄고 같은 질문을 반복한다**

`backend/.env`에 `RERANK_ENABLED=false`를 넣고 서버를 재시작한 뒤 **같은 질문**을 던진다.

- [ ] **Step 5: 관찰을 기록한다**

`docs/troubleshooting/2026-09-03-reranking-ab.md`를 만들어 다음을 남긴다.

- 출처 목록이 어떻게 달라졌는지 — 리랭킹으로 새로 올라온 청크가 실제로 더 관련 있었는지
- 답변이 달라졌는지
- 질의당 지연 차이
- 첫 질의의 모델 로딩 시간

기대 효과는 "답이 정확해진다"가 아니라 **"벡터 검색이 밀어낸 청크가 구제되는 사례가 생긴다"** 이다.

**개선이 관찰되지 않으면** `config.py`의 `rerank_enabled` 기본값을 `False`로 바꾸고, 그 판단 근거를 같은 문서에 적는다.

- [ ] **Step 6: 커밋한다**

```bash
git add docs/troubleshooting/2026-09-03-reranking-ab.md
git commit -m "docs: record the reranking A/B observations"
```
