from langchain.schema import Document

from app.services.rag_service import RAGService


def _format(docs):
    """RAGService를 생성하지 않고 포맷터만 호출한다.

    __init__이 임베딩 모델과 Ollama 연결을 요구하므로 우회한다.
    """
    service = RAGService.__new__(RAGService)
    return service._format_sources(docs)


def test_chunk_with_image_ids_gets_relative_image_urls():
    doc = Document(
        page_content="[이미지 텍스트]\n생활SOC 확충",
        metadata={
            "filename": "고시.pdf", "document_id": "doc_abc123",
            "page": 1, "chunk_index": 7,
            # Chroma metadata can only hold scalars, so image_ids is stored
            # (and comes back from retrieval) as a comma-joined string.
            "image_ids": "p2_a1b2c3,p2_full",
        },
    )

    sources = _format([doc])

    assert sources[0].image_urls == [
        "/api/v1/documents/doc_abc123/images/p2_a1b2c3",
        "/api/v1/documents/doc_abc123/images/p2_full",
    ]


def test_chunk_without_image_ids_has_empty_list():
    doc = Document(
        page_content="본문 텍스트",
        metadata={"filename": "고시.pdf", "document_id": "doc_abc123", "chunk_index": 0},
    )

    sources = _format([doc])

    assert sources[0].image_urls == []


def test_missing_document_id_yields_no_urls_even_with_image_ids():
    """document_id가 없으면 깨진 링크를 만들지 않는다."""
    doc = Document(
        page_content="텍스트",
        metadata={"filename": "고시.pdf", "chunk_index": 0, "image_ids": "p0_x"},
    )

    sources = _format([doc])

    assert sources[0].image_urls == []


def test_similarity_score_is_read_from_metadata():
    """ScoringRetriever가 채워 넣은 점수를 그대로 응답에 실어 보낸다."""
    doc = Document(
        page_content="텍스트",
        metadata={"filename": "고시.pdf", "chunk_index": 0, "similarity_score": 0.8321},
    )

    sources = _format([doc])

    assert sources[0].similarity_score == 0.8321


def test_missing_similarity_score_is_none_not_zero():
    """점수가 없을 때 0.0으로 오해되면 프런트에서 'Relevance: 0.0%'로 잘못 표시된다."""
    doc = Document(
        page_content="텍스트",
        metadata={"filename": "고시.pdf", "chunk_index": 0},
    )

    sources = _format([doc])

    assert sources[0].similarity_score is None
