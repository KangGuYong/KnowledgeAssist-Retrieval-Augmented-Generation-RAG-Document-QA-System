"""Chunking strategy factory: "default" (character-based) vs "semantic"."""

import pytest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.embeddings import FakeEmbeddings

from app.services.chunking import build_splitter


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown_strategy"):
        build_splitter("unknown_strategy")


def test_chunking_settings_default_to_the_character_splitter():
    from app.config import Settings

    settings = Settings()

    assert settings.chunking_strategy == "default"
    assert settings.semantic_chunker_breakpoint_type == "percentile"


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
    from langchain_experimental.text_splitter import SemanticChunker

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
    # SemanticChunker.split_documents() carries the parent metadata onto every
    # chunk (verified against the actual 0.0.55 source) - this is what keeps
    # the OCR image citation feature (image_ids, page, ...) working no matter
    # which chunking strategy is chosen.
    assert all(c.metadata.get("filename") == "고시.pdf" for c in chunks)
    assert all(c.metadata.get("page") == 2 for c in chunks)
