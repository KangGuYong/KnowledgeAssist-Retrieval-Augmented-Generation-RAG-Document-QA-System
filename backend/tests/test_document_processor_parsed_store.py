"""document_id가 주어지면 load_pdf()가 parsed_store를 통해 MinerU 원본
블록을 JSON으로 남기는지, 안 주어지면 남기지 않는지 검증한다."""
import json

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
    import base64
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _client_with_text_and_image():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [
        {"type": "text", "page_idx": 0, "text": "Native text before the diagram"},
        {"type": "image", "page_idx": 0, "img_path": "images/img1.png"},
    ]
    return FakeMineruClient(blocks, images)


def _patch_storage_dirs(monkeypatch, tmp_path):
    from app.services import document_processor as document_processor_module
    from app.services import parsed_store as parsed_store_module

    monkeypatch.setattr(
        document_processor_module.settings, "image_storage_dir", str(tmp_path / "images")
    )
    monkeypatch.setattr(
        parsed_store_module.settings, "parsed_storage_dir", str(tmp_path / "parsed")
    )
    return tmp_path


def test_parsed_json_is_written_when_document_id_given(tmp_path, monkeypatch):
    tmp_path = _patch_storage_dirs(monkeypatch, tmp_path)
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image()
    )

    processor.process_file("ignored.pdf", "diagram.pdf", document_id="doc_parsed_test")

    parsed_path = tmp_path / "parsed" / "doc_parsed_test.json"
    assert parsed_path.exists()
    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    assert data["filename"] == "diagram.pdf"
    texts = [b.get("text") for b in data["pages"][0]["blocks"] if b.get("text")]
    assert "Native text before the diagram" in texts


def test_parsed_json_is_not_written_when_document_id_missing(tmp_path, monkeypatch):
    tmp_path = _patch_storage_dirs(monkeypatch, tmp_path)
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image()
    )

    processor.process_file("ignored.pdf", "diagram.pdf")

    assert not (tmp_path / "parsed").exists()


def test_mineru_disabled_skips_parsed_json_too(tmp_path, monkeypatch):
    tmp_path = _patch_storage_dirs(monkeypatch, tmp_path)
    from app.services import document_processor as module

    monkeypatch.setattr(module.settings, "mineru_enabled", False)
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Native text before the diagram", fontsize=12)
    pdf_path = tmp_path / "diagram.pdf"
    doc.save(pdf_path)
    doc.close()

    processor = DocumentProcessor(
        ocr=StubOCR("무시되어야 함"), mineru_client=_client_with_text_and_image()
    )

    processor.process_file(str(pdf_path), "diagram.pdf", document_id="doc_disabled_test")

    assert not (tmp_path / "parsed").exists()


def test_chunks_are_unaffected_by_parsed_store_persistence(tmp_path, monkeypatch):
    tmp_path = _patch_storage_dirs(monkeypatch, tmp_path)
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image()
    )

    chunks = processor.process_file("ignored.pdf", "diagram.pdf", document_id="doc_regression_test")

    joined = "\n".join(c.page_content for c in chunks)
    assert "Native text before the diagram" in joined
    assert "그림 안의 설명 문장" in joined
