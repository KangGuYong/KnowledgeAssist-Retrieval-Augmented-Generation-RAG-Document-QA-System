"""ScoringRetriever must expose the similarity score that as_retriever() drops."""

import pytest
from langchain_core.documents import Document

from app.services.rag_service import ScoringRetriever


class FakeVectorStore:
    """Stands in for Chroma: records how it was called, returns canned results."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.results


def test_relevance_score_is_stashed_in_document_metadata():
    store = FakeVectorStore([
        (Document(page_content="a", metadata={}), 0.91),
        (Document(page_content="b", metadata={}), 0.42),
    ])
    # 이 테스트의 관심사는 점수 부착이지 순서가 아니다. 재배치를 끄고 격리한다.
    retriever = ScoringRetriever(vector_store=store, k=4, reorder=False)

    docs = retriever.invoke("생활SOC 확충 계획")

    assert [d.metadata["similarity_score"] for d in docs] == [0.91, 0.42]
    assert [d.page_content for d in docs] == ["a", "b"]


def test_query_k_and_filter_are_forwarded_to_the_vector_store():
    store = FakeVectorStore([])
    retriever = ScoringRetriever(
        vector_store=store, k=7, search_filter={"document_id": {"$in": ["doc_1"]}}
    )

    retriever.invoke("질문")

    assert store.calls == [
        {"query": "질문", "k": 7, "filter": {"document_id": {"$in": ["doc_1"]}}}
    ]


def test_no_filter_passes_none_through():
    store = FakeVectorStore([])
    retriever = ScoringRetriever(vector_store=store, k=4)

    retriever.invoke("질문")

    assert store.calls[0]["filter"] is None


@pytest.mark.parametrize(
    "k, expected",
    [
        # 배포 설정값(backend/.env의 RETRIEVAL_K=5). 홀수라 1등이 맨 앞에 온다.
        (5, ["1", "3", "5", "4", "2"]),
        # 클래스 기본값. 짝수라 방향이 뒤집혀 1등이 맨 뒤로 간다.
        (10, ["2", "4", "6", "8", "10", "9", "7", "5", "3", "1"]),
    ],
)
def test_reordering_moves_the_best_chunks_to_the_edges(k, expected):
    """기본값(reorder=True)에서 관련성 높은 청크가 컨텍스트 양 끝으로 간다.

    방향은 k의 홀짝에 따라 뒤집히지만(LongContextReorder의 동작,
    tests/test_long_context_reorder.py가 계약으로 고정한다) 양 끝에 상위 청크가
    온다는 성질은 어느 쪽이든 유지된다. 배포값 5와 클래스 기본값 10을 모두 건다.
    """
    store = FakeVectorStore([
        (Document(page_content=str(i + 1), metadata={}), 1.0 - i * 0.01)
        for i in range(k)
    ])
    retriever = ScoringRetriever(vector_store=store, k=k)

    docs = retriever.invoke("질문")

    assert [d.page_content for d in docs] == expected


def test_scores_survive_the_reordering():
    """재배치 후에도 점수가 각 문서를 따라다녀야 출처 정렬이 가능하다."""
    store = FakeVectorStore([
        (Document(page_content="a", metadata={}), 0.91),
        (Document(page_content="b", metadata={}), 0.42),
        (Document(page_content="c", metadata={}), 0.30),
    ])
    retriever = ScoringRetriever(vector_store=store, k=3)

    docs = retriever.invoke("질문")
    by_content = {d.page_content: d.metadata["similarity_score"] for d in docs}

    assert by_content == {"a": 0.91, "b": 0.42, "c": 0.30}


def test_reorder_can_be_turned_off():
    store = FakeVectorStore([
        (Document(page_content="a", metadata={}), 0.91),
        (Document(page_content="b", metadata={}), 0.42),
    ])
    retriever = ScoringRetriever(vector_store=store, k=2, reorder=False)

    docs = retriever.invoke("질문")

    assert [d.page_content for d in docs] == ["a", "b"]


def test_empty_search_result_does_not_blow_up():
    retriever = ScoringRetriever(vector_store=FakeVectorStore([]), k=10)

    assert retriever.invoke("질문") == []


@pytest.mark.parametrize("configured", [True, False])
def test_ask_question_injects_the_reorder_setting_into_the_retriever(monkeypatch, configured):
    """RETRIEVAL_REORDER가 실제로 리트리버까지 전달되는지 확인한다.

    한쪽 방향만 검사하면 reorder를 상수로 박아넣은 코드를 잡지 못한다.
    """
    import asyncio
    from types import SimpleNamespace

    import app.services.rag_service as rag_module
    from app.services.rag_service import RAGService

    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    captured = {}
    real_retriever_cls = rag_module.ScoringRetriever

    def spy(**kwargs):
        # 진짜 리트리버를 만들되 참조를 붙잡아 둔다. 대역으로 갈아끼우면
        # 필드가 실제로 유효한 값인지는 검사하지 못한다.
        captured["retriever"] = real_retriever_cls(**kwargs)
        return captured["retriever"]

    monkeypatch.setattr(rag_module, "ScoringRetriever", spy)
    # 실제 Settings 대신 필요한 두 값만 가진 대역으로 갈아끼운다.
    monkeypatch.setattr(
        rag_module, "settings",
        SimpleNamespace(retrieval_k=10, retrieval_reorder=configured),
    )

    # __init__은 임베딩 모델과 Ollama 연결을 요구하므로 우회한다.
    service = RAGService.__new__(RAGService)
    service.vector_store = FakeVectorStore([])
    service.llm = RunnableLambda(lambda _: AIMessage(content="ok"))
    service.conversation_histories = {}

    asyncio.run(service.ask_question("질문"))

    assert captured["retriever"].reorder is configured
    assert captured["retriever"].k == 10
