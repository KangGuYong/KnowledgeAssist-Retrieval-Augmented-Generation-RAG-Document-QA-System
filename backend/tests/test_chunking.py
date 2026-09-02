"""Chunking strategy factory: "default" (character-based) vs "semantic".

The semantic path is a direct port of Greg Kamradt's "5 Levels of Text
Splitting" notebook (Level 4: Semantic Chunking) - see chunking.py's module
docstring for the source link. These tests verify that port against the
notebook's own algorithm, not just "does it run".
"""

import pytest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.embeddings import FakeEmbeddings

from app.services.chunking import SemanticChunker, _combine_sentences, _cosine_distances, build_splitter


class _FixedEmbeddings:
    """Returns pre-chosen vectors, one per call, in the order embed_documents receives them."""

    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, texts):
        assert len(texts) == len(self.vectors), "test wired up the wrong number of vectors"
        return self.vectors


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown_strategy"):
        build_splitter("unknown_strategy")


def test_chunking_settings_default_to_the_character_splitter():
    from app.config import Settings

    settings = Settings()

    assert settings.chunking_strategy == "default"
    assert settings.semantic_chunker_breakpoint_percentile == 95.0


def test_default_strategy_returns_recursive_character_splitter():
    splitter = build_splitter("default")

    assert isinstance(splitter, RecursiveCharacterTextSplitter)


def test_default_strategy_honours_explicit_chunk_size_and_overlap():
    splitter = build_splitter("default", chunk_size=333, chunk_overlap=55)

    assert splitter._chunk_size == 333
    assert splitter._chunk_overlap == 55


def test_default_strategy_falls_back_to_settings_when_not_given():
    from app.config import get_settings

    settings = get_settings()
    splitter = build_splitter("default")

    assert splitter._chunk_size == settings.chunk_size
    assert splitter._chunk_overlap == settings.chunk_overlap


def test_semantic_strategy_uses_the_injected_embeddings_not_a_real_model():
    """실제 KURE 모델을 로드하지 않고도 테스트가 빨리 끝나야 한다."""
    fake = FakeEmbeddings(size=8)
    splitter = build_splitter("semantic", embeddings=fake)

    assert isinstance(splitter, SemanticChunker)
    assert splitter.embeddings is fake


def test_semantic_strategy_splits_documents_without_losing_content():
    fake = FakeEmbeddings(size=8)
    splitter = build_splitter("semantic", embeddings=fake)
    doc = Document(
        page_content=(
            "분당신도시 노후계획도시정비기본계획의 목표는 생활SOC 확충이다. "
            "야탑역 인근 지역문화복지시설을 신설한다. "
            "율동공원 일대의 녹지축은 그대로 보존한다. "
            "탄천 수변공간은 시민 접근성을 높이는 방향으로 재정비한다."
        ),
        metadata={"filename": "고시.pdf", "page": 2},
    )

    chunks = splitter.split_documents([doc])

    assert chunks
    joined = " ".join(c.page_content for c in chunks)
    assert "생활SOC 확충" in joined
    assert "탄천 수변공간" in joined
    # split_documents() carries the parent metadata onto every chunk - this is
    # what keeps the OCR image citation feature (image_ids, page, ...) working
    # no matter which chunking strategy is chosen.
    assert all(c.metadata.get("filename") == "고시.pdf" for c in chunks)
    assert all(c.metadata.get("page") == 2 for c in chunks)


# --- Algorithm-level tests against the notebook's own logic --------------


def test_combine_sentences_windows_one_neighbour_each_side_by_default():
    """notebook cell 84: buffer_size=1 combines [prev, current, next]."""
    sentences = [{"sentence": s, "index": i} for i, s in enumerate(["A.", "B.", "C."])]

    combined = _combine_sentences(sentences, buffer_size=1)

    assert combined[0]["combined_sentence"] == "A. B."
    assert combined[1]["combined_sentence"] == "A. B. C."
    assert combined[2]["combined_sentence"] == "B. C."


def test_cosine_distance_of_identical_vectors_is_zero():
    assert _cosine_distances([[1, 0], [1, 0]]) == pytest.approx([0.0])


def test_cosine_distance_of_orthogonal_vectors_is_one():
    assert _cosine_distances([[1, 0], [0, 1]]) == pytest.approx([1.0])


def test_split_text_breaks_at_the_largest_semantic_distance():
    """3 sentences, buffer_size=1 -> 2 consecutive-distance values.

    Vectors are chosen so combined_sentence[0] and [1] point nearly the same
    way (topic: pets) and [2] points orthogonally (topic: stock market) -
    exactly the "find break points between sequential sentences" method the
    notebook describes.
    """
    embeddings = _FixedEmbeddings([[1, 0], [1, 0.05], [0, 1]])
    splitter = SemanticChunker(embeddings, buffer_size=1, breakpoint_percentile=95.0)

    chunks = splitter.split_text(
        "Cats are great pets. Dogs are great pets too. The stock market crashed today."
    )

    assert chunks == [
        "Cats are great pets. Dogs are great pets too.",
        "The stock market crashed today.",
    ]


def test_split_text_with_a_single_sentence_returns_it_unsplit():
    """np.percentile on an empty distances list would raise - guard it."""
    splitter = SemanticChunker(FakeEmbeddings(size=4))

    assert splitter.split_text("혼자인 문장.") == ["혼자인 문장."]


# --- Chunks must not be trapped inside one page --------------------------
#
# A 545-page large-print PDF held ~200 characters per page. Because both
# splitters work one Document at a time and load_pdf() emits one Document per
# page, chunk_size=1000 was never reachable: the median chunk came out at 158
# characters, and the semantic percentile was being computed over the two or
# three sentence distances a single page happened to contain.


def _page(text, page_number, **metadata):
    return Document(
        page_content=text,
        metadata={
            "page": page_number - 1,
            "page_number": page_number,
            "source": "/tmp/book.pdf",
            "filename": "book.pdf",
            **metadata,
        },
    )


def test_merge_pages_combines_small_pages_up_to_the_target_size():
    from app.services.chunking import merge_pages

    pages = [_page("가" * 100, i) for i in range(1, 11)]

    merged = merge_pages(pages, target_size=400)

    assert len(merged) < len(pages)
    assert all(len(m.page_content) >= 400 for m in merged[:-1])


def test_merge_pages_keeps_the_first_pages_metadata_and_records_the_span():
    from app.services.chunking import merge_pages

    pages = [_page("나" * 100, i) for i in range(3, 8)]

    merged = merge_pages(pages, target_size=400)

    assert merged[0].metadata["page_number"] == 3
    assert merged[0].metadata["page"] == 2
    assert merged[0].metadata["page_number_end"] == 6


def test_merge_pages_unions_image_ids_and_ocr_flags_across_merged_pages():
    """Source citations render images from image_ids, so merging pages must
    carry every merged page's images rather than only the first page's."""
    from app.services.chunking import merge_pages

    pages = [
        _page("다" * 100, 1, image_ids="aaa", ocr_used=True, ocr_image_count=1),
        _page("라" * 100, 2, image_ids="bbb,ccc", ocr_used=False, ocr_image_count=0),
    ]

    merged = merge_pages(pages, target_size=1000)

    assert len(merged) == 1
    assert merged[0].metadata["image_ids"] == "aaa,bbb,ccc"
    assert merged[0].metadata["ocr_used"] is True
    assert merged[0].metadata["ocr_image_count"] == 1


def test_merge_pages_never_merges_across_two_source_files():
    from app.services.chunking import merge_pages

    pages = [
        _page("마" * 50, 1, source="/tmp/a.pdf"),
        _page("바" * 50, 1, source="/tmp/b.pdf"),
    ]

    merged = merge_pages(pages, target_size=1000)

    assert len(merged) == 2


def test_merge_pages_leaves_pages_that_already_reach_the_target_alone():
    from app.services.chunking import merge_pages

    pages = [_page("사" * 1200, i) for i in range(1, 4)]

    merged = merge_pages(pages, target_size=1000)

    assert len(merged) == 3


def test_semantic_breakpoint_threshold_is_computed_across_all_documents():
    """Per-document percentiles are meaningless on a page holding two or three
    sentences - the threshold is relative, so a page with no topic shift gets
    split anyway. One threshold over the whole document fixes that."""
    calm = Document(page_content="가 하나. 가 둘. 가 셋.", metadata={"page": 0})
    shifting = Document(page_content="나 하나. 나 둘. 나 셋.", metadata={"page": 1})
    # calm's largest gap is small; shifting's last gap is orthogonal. A global
    # 95th percentile sits above calm's gaps, so only shifting breaks.
    embeddings = _FixedEmbeddings(
        [[1, 0], [1, 0.01], [1, 0.5]] + [[1, 0], [1, 0], [0, 1]]
    )
    splitter = SemanticChunker(embeddings, buffer_size=1, breakpoint_percentile=95.0)

    chunks = splitter.split_documents([calm, shifting])

    from_calm = [c for c in chunks if c.metadata.get("page") == 0]
    from_shifting = [c for c in chunks if c.metadata.get("page") == 1]
    assert len(from_calm) == 1, "no topic shift here - must not be split"
    assert len(from_shifting) == 2


def test_semantic_chunker_does_not_emit_chunks_below_the_minimum_size():
    """192 of the 1081 stored chunks were under 50 characters and five held
    nothing but a page number."""
    # Distances 1.0 and ~0.9 against a 50th-percentile threshold of ~0.95:
    # the first gap is a breakpoint, but honouring it would emit a 3-character
    # chunk, so it has to be carried into the next one instead.
    embeddings = _FixedEmbeddings([[1, 0], [0, 1], [1, 0.1]])
    splitter = SemanticChunker(
        embeddings, buffer_size=1, breakpoint_percentile=50.0, min_chunk_chars=200
    )

    chunks = splitter.split_text("짧다. 또 짧다. 여전히 짧다.")

    assert len(chunks) == 1


def test_min_chunk_chars_defaults_to_zero_so_the_notebook_port_is_unchanged():
    splitter = SemanticChunker(FakeEmbeddings(size=4))

    assert splitter.min_chunk_chars == 0


def test_semantic_splitter_from_the_factory_uses_the_configured_minimum():
    from app.config import get_settings

    splitter = build_splitter("semantic", embeddings=FakeEmbeddings(size=8))

    assert splitter.min_chunk_chars == get_settings().semantic_chunker_min_chunk_chars


def test_chunk_documents_merges_tiny_pages_before_splitting():
    """End to end: 40 pages of ~60 characters must not yield 40 tiny chunks."""
    from app.services.document_processor import DocumentProcessor

    pages = [_page("가나다라마바사아자차. " * 5, i) for i in range(1, 41)]

    chunks = DocumentProcessor().chunk_documents(
        pages, "book.pdf", chunking_strategy="default", chunk_size=1000, chunk_overlap=100
    )

    assert len(chunks) < len(pages)
    assert max(len(c.page_content) for c in chunks) > 500
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["filename"] == "book.pdf"
