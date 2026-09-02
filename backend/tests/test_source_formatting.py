from langchain_core.documents import Document

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


def test_negative_scores_still_sort_above_a_document_with_no_score():
    """관련성 점수는 음수가 될 수 있다.

    Chroma 컬렉션이 hnsw:space 없이 만들어져 LangChain이 유클리드 변환식을
    쓰므로 점수가 0 아래로 내려간다. 점수 없는 문서를 0.0으로 취급하면 음수
    점수를 가진 진짜 출처보다 위로 올라가 버린다.
    """
    docs = [
        Document(page_content="점수 없음", metadata={
            "filename": "a.pdf", "document_id": "doc_1", "chunk_index": 0,
        }),
        Document(page_content="음수 점수", metadata={
            "filename": "a.pdf", "document_id": "doc_1",
            "chunk_index": 1, "similarity_score": -0.2,
        }),
    ]

    sources = _format(docs)

    assert [s.content for s in sources] == ["음수 점수", "점수 없음"]
