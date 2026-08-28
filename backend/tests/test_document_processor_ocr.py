"""PDF text recovered from images must reach the chunks that get embedded."""

import fitz

from app.services.document_processor import DocumentProcessor


class StubOCR:
    def __init__(self, text):
        self.text = text

    def image_to_text(self, image):
        return self.text


def _pdf_with_image(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Native text before the diagram", fontsize=12)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 90))
    pixmap.set_rect(pixmap.irect, (10, 90, 200))
    page.insert_image(fitz.Rect(72, 200, 312, 380), stream=pixmap.tobytes("png"))
    path = tmp_path / "diagram.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def test_chunks_contain_the_text_recovered_from_images(tmp_path):
    processor = DocumentProcessor(ocr=StubOCR("그림 안의 설명 문장"))

    chunks = processor.process_file(_pdf_with_image(tmp_path), "diagram.pdf")

    assert chunks
    joined = "\n".join(c.page_content for c in chunks)
    assert "Native text before the diagram" in joined
    assert "그림 안의 설명 문장" in joined


def test_chunk_metadata_records_the_ocr_provenance(tmp_path):
    processor = DocumentProcessor(ocr=StubOCR("표 안의 숫자"))

    chunks = processor.process_file(_pdf_with_image(tmp_path), "diagram.pdf")

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
    processor = DocumentProcessor(ocr=StubOCR("무시되어야 하는 텍스트"))

    chunks = processor.process_file(_pdf_with_image(tmp_path), "diagram.pdf")

    joined = "\n".join(c.page_content for c in chunks)
    assert "무시되어야 하는 텍스트" not in joined
    assert "Native text before the diagram" in joined


def test_image_ids_are_carried_onto_every_chunk_from_that_page(tmp_path, monkeypatch):
    """청크-이미지 연결은 페이지 단위 근사다(설계 문서 3.1절):
    그 페이지에서 나온 모든 청크가 그 페이지의 이미지 전부를 인용한다."""
    from app.services import document_processor as module

    monkeypatch.setattr(module.settings, "image_storage_dir", str(tmp_path / "images"))
    processor = DocumentProcessor(ocr=StubOCR("그림 안의 설명 문장"))

    chunks = processor.process_file(
        _pdf_with_image(tmp_path), "diagram.pdf", document_id="doc_test123"
    )

    assert chunks
    assert all(c.metadata["image_ids"] for c in chunks)
    # Chroma metadata can only hold scalars (str/int/float/bool); a list value
    # makes add_documents() raise "Expected metadata value to be a str, int,
    # float or bool". image_ids must be a comma-joined string, not a list.
    assert all(isinstance(c.metadata["image_ids"], str) for c in chunks)
    image_id = chunks[0].metadata["image_ids"].split(",")[0]
    assert (tmp_path / "images" / "doc_test123" / f"{image_id}.png").exists()


def test_no_document_id_skips_image_saving(tmp_path):
    """document_id 없이 호출하는 기존 방식(테스트 등)은 그대로 동작해야 한다."""
    processor = DocumentProcessor(ocr=StubOCR("그림 안의 설명 문장"))

    chunks = processor.process_file(_pdf_with_image(tmp_path), "diagram.pdf")

    assert chunks
    assert all(c.metadata.get("image_ids", "") == "" for c in chunks)
