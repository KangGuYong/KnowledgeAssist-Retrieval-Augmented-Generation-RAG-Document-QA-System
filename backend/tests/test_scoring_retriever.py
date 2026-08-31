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
    retriever = ScoringRetriever(vector_store=store, k=4)

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
