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
