"""PDF text recovered from MinerU + image OCR must reach the chunks that get embedded."""

from pathlib import Path

import fitz

from app.services.document_processor import DocumentProcessor
from app.services.mineru_client import MineruResult


class StubOCR:
    def __init__(self, text):
        self.text = text

    def image_to_text(self, image):
        return self.text


class FakeMineruClient:
    def __init__(self, blocks, images):
        self.blocks = blocks
        self.images = images

    def parse_pdf(self, file_path):
        return MineruResult(blocks=self.blocks, images=self.images)


def _png_data_uri(color=(10, 90, 200), size=(20, 15)):
    """MinerU's /file_parse images dict value shape: a base64 data URI."""
    import base64
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _client_with_text_and_image(tmp_path=None):
    """text 블록 하나 + image 블록 하나짜리 가짜 MinerU 응답. tmp_path 인자는
    다른 태스크의 호출부와 시그니처를 맞추기 위해 남겨둔다(사용하지 않음 -
    이미지는 더 이상 디스크 파일이 아니라 base64 데이터 URI로 인라인된다)."""
    images = {"images/img1.png": _png_data_uri()}
    blocks = [
        {"type": "text", "page_idx": 0, "text": "Native text before the diagram"},
        {"type": "image", "page_idx": 0, "img_path": "images/img1.png"},
    ]
    return FakeMineruClient(blocks, images)


def _real_minimal_pdf(tmp_path, name="diagram.pdf"):
    """MinerU 실패 폴백(PyPDFLoader) 테스트용 - 이 한 곳만 실제 파일이 필요하다."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Native text before the diagram", fontsize=12)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return str(path)


def test_chunks_contain_the_text_recovered_from_images(tmp_path):
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf")

    assert chunks
    joined = "\n".join(c.page_content for c in chunks)
    assert "Native text before the diagram" in joined
    assert "그림 안의 설명 문장" in joined


def test_chunk_metadata_records_the_ocr_provenance(tmp_path):
    processor = DocumentProcessor(
        ocr=StubOCR("표 안의 숫자"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf")

    metadata = chunks[0].metadata
    assert metadata["filename"] == "diagram.pdf"
    assert metadata["page"] == 0
    assert metadata["page_number"] == 1
    assert metadata["ocr_used"] is True
    assert metadata["ocr_image_count"] == 1
    assert metadata["chunk_index"] == 0


def test_ocr_can_be_turned_off(tmp_path, monkeypatch):
    from app.services import document_processor as module

    monkeypatch.setattr(module.settings, "ocr_enabled", False)
    processor = DocumentProcessor(
        ocr=StubOCR("무시되어야 하는 텍스트"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf")

    joined = "\n".join(c.page_content for c in chunks)
    assert "무시되어야 하는 텍스트" not in joined
    assert "Native text before the diagram" in joined


def test_mineru_disabled_falls_back_to_plain_text(tmp_path, monkeypatch):
    from app.services import document_processor as module

    monkeypatch.setattr(module.settings, "mineru_enabled", False)
    processor = DocumentProcessor(
        ocr=StubOCR("이 파이프라인은 호출되지 않아야 한다"),
        mineru_client=_client_with_text_and_image(tmp_path),
    )

    chunks = processor.process_file(_real_minimal_pdf(tmp_path), "diagram.pdf")

    joined = "\n".join(c.page_content for c in chunks)
    assert "Native text before the diagram" in joined
    assert "이 파이프라인은 호출되지 않아야 한다" not in joined


def test_mineru_failure_falls_back_to_plain_text(tmp_path):
    class FailingMineruClient:
        def parse_pdf(self, file_path):
            raise ConnectionError("MinerU service unreachable")

    processor = DocumentProcessor(ocr=StubOCR("도달하지 않아야 함"), mineru_client=FailingMineruClient())

    chunks = processor.process_file(_real_minimal_pdf(tmp_path), "diagram.pdf")

    joined = "\n".join(c.page_content for c in chunks)
    assert "Native text before the diagram" in joined
    assert "도달하지 않아야 함" not in joined


def test_image_ids_are_carried_onto_every_chunk_from_that_page(tmp_path, monkeypatch):
    """청크-이미지 연결은 페이지 단위 근사다(설계 문서 3.1절, 2026-08-28):
    그 페이지에서 나온 모든 청크가 그 페이지의 이미지 전부를 인용한다."""
    from app.services import document_processor as module
    from app.services import parsed_store as parsed_store_module

    monkeypatch.setattr(module.settings, "image_storage_dir", str(tmp_path / "images_out"))
    monkeypatch.setattr(parsed_store_module.settings, "parsed_storage_dir", str(tmp_path / "parsed_out"))
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf", document_id="doc_test123")

    assert chunks
    assert all(c.metadata["image_ids"] for c in chunks)
    # Chroma metadata can only hold scalars (str/int/float/bool); a list value
    # makes add_documents() raise "Expected metadata value to be a str, int,
    # float or bool". image_ids must be a comma-joined string, not a list.
    assert all(isinstance(c.metadata["image_ids"], str) for c in chunks)
    image_id = chunks[0].metadata["image_ids"].split(",")[0]
    assert (tmp_path / "images_out" / "doc_test123" / f"{image_id}.png").exists()


def test_no_document_id_skips_image_saving(tmp_path):
    """document_id 없이 호출하는 기존 방식(테스트 등)은 그대로 동작해야 한다."""
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf")

    assert chunks
    assert all(c.metadata.get("image_ids", "") == "" for c in chunks)


def test_default_chunking_strategy_used_when_not_specified(tmp_path):
    """기존 호출부(인자 없이 process_file)는 그대로 'default' 전략으로 동작해야 한다."""
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf")

    assert chunks
    assert all(c.metadata["chunking_strategy"] == "default" for c in chunks)
    assert all("chunk_size" in c.metadata for c in chunks)
    assert all("chunk_overlap" in c.metadata for c in chunks)


def test_explicit_chunk_size_and_overlap_are_recorded_in_metadata(tmp_path):
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file(
        "ignored.pdf", "diagram.pdf",
        chunking_strategy="default", chunk_size=333, chunk_overlap=55,
    )

    assert chunks
    assert all(c.metadata["chunk_size"] == 333 for c in chunks)
    assert all(c.metadata["chunk_overlap"] == 55 for c in chunks)


class _FakeSplitter:
    """RecursiveCharacterTextSplitter/SemanticChunker의 split_documents 계약만 흉내낸다."""

    def split_documents(self, documents):
        return list(documents)


def test_semantic_strategy_does_not_write_chunk_size_or_overlap_metadata(tmp_path, monkeypatch):
    """Chroma 메타데이터는 스칼라만 허용한다(이미 image_ids에서 겪은 문제 - 커밋 af38b9f).
    시멘틱 전략에는 chunk_size/overlap 개념이 없으므로 None을 쓰는 대신 키 자체를 뺀다."""
    from app.services import document_processor as module

    monkeypatch.setattr(module, "build_splitter", lambda *a, **k: _FakeSplitter())
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )

    chunks = processor.process_file(
        "ignored.pdf", "diagram.pdf", chunking_strategy="semantic",
    )

    assert chunks
    assert all(c.metadata["chunking_strategy"] == "semantic" for c in chunks)
    assert all("chunk_size" not in c.metadata for c in chunks)
    assert all("chunk_overlap" not in c.metadata for c in chunks)
