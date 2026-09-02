"""ScoringRetriever must expose the similarity score that as_retriever() drops."""

from langchain.schema import Document

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

    docs = retriever.get_relevant_documents("생활SOC 확충 계획")

    assert [d.metadata["similarity_score"] for d in docs] == [0.91, 0.42]
    assert [d.page_content for d in docs] == ["a", "b"]


def test_query_k_and_filter_are_forwarded_to_the_vector_store():
    store = FakeVectorStore([])
    retriever = ScoringRetriever(
        vector_store=store, k=7, search_filter={"document_id": {"$in": ["doc_1"]}}
    )

    retriever.get_relevant_documents("질문")

    assert store.calls == [
        {"query": "질문", "k": 7, "filter": {"document_id": {"$in": ["doc_1"]}}}
    ]


def test_no_filter_passes_none_through():
    store = FakeVectorStore([])
    retriever = ScoringRetriever(vector_store=store, k=4)

    retriever.get_relevant_documents("질문")

    assert store.calls[0]["filter"] is None


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
