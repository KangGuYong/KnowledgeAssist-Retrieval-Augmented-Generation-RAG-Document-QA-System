"""parsed_store.save()가 MinerU 원본 블록을 문서별 JSON으로 정확히
영속화하는지, 그리고 어떤 실패에도 예외를 전파하지 않는지 검증한다."""
import json

from app.services import parsed_store
from app.services.mineru_client import MineruResult


def _png_data_uri(color=(10, 90, 200), size=(20, 15)):
    import base64
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_save_writes_json_grouped_by_page(tmp_path, monkeypatch):
    monkeypatch.setattr(parsed_store.settings, "parsed_storage_dir", str(tmp_path / "parsed"))
    result = MineruResult(
        blocks=[
            {"type": "title", "page_idx": 0, "text": "제목"},
            {"type": "text", "page_idx": 0, "text": "본문"},
            {"type": "text", "page_idx": 1, "text": "둘째 페이지"},
        ],
        images={},
    )

    parsed_store.save("doc_a", "a.pdf", result, tmp_path / "images")

    data = json.loads((tmp_path / "parsed" / "doc_a.json").read_text(encoding="utf-8"))
    assert data["document_id"] == "doc_a"
    assert data["filename"] == "a.pdf"
    assert data["page_count"] == 2
    assert [p["page_number"] for p in data["pages"]] == [1, 2]
    assert data["pages"][0]["blocks"] == [
        {"type": "title", "text": "제목"},
        {"type": "text", "text": "본문"},
    ]
    assert data["pages"][1]["blocks"] == [{"type": "text", "text": "둘째 페이지"}]


def test_save_persists_image_blocks_and_records_image_id(tmp_path, monkeypatch):
    monkeypatch.setattr(parsed_store.settings, "parsed_storage_dir", str(tmp_path / "parsed"))
    data_uri = _png_data_uri()
    result = MineruResult(
        blocks=[{"type": "image", "page_idx": 0, "img_path": "images/fig1.png"}],
        images={"images/fig1.png": data_uri},
    )
    image_dir = tmp_path / "images" / "doc_b"

    parsed_store.save("doc_b", "b.pdf", result, image_dir)

    data = json.loads((tmp_path / "parsed" / "doc_b.json").read_text(encoding="utf-8"))
    block = data["pages"][0]["blocks"][0]
    assert block["type"] == "image"
    assert "image_id" in block
    assert (image_dir / f"{block['image_id']}.png").exists()


def test_save_keeps_table_body_and_image_id_together_when_both_present(tmp_path, monkeypatch):
    """표 블록은 table_body(HTML)와 img_path(스크린샷)를 동시에 가질 수
    있다 - 뷰어는 둘 다, 원본 그대로 보여줘야 한다(design doc 3.4절)."""
    monkeypatch.setattr(parsed_store.settings, "parsed_storage_dir", str(tmp_path / "parsed"))
    data_uri = _png_data_uri()
    result = MineruResult(
        blocks=[
            {
                "type": "table",
                "page_idx": 0,
                "img_path": "images/table1.png",
                "table_body": "<table><tr><td>120억</td></tr></table>",
            }
        ],
        images={"images/table1.png": data_uri},
    )

    parsed_store.save("doc_c", "c.pdf", result, tmp_path / "images" / "doc_c")

    data = json.loads((tmp_path / "parsed" / "doc_c.json").read_text(encoding="utf-8"))
    block = data["pages"][0]["blocks"][0]
    assert block["table_body"] == "<table><tr><td>120억</td></tr></table>"
    assert "image_id" in block


def test_save_never_raises_when_persisting_fails(tmp_path, monkeypatch):
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("blocks parsed_storage_dir from being a directory")
    monkeypatch.setattr(parsed_store.settings, "parsed_storage_dir", str(blocking_file))
    result = MineruResult(blocks=[{"type": "text", "page_idx": 0, "text": "x"}], images={})

    parsed_store.save("doc_d", "d.pdf", result, tmp_path / "images")  # must not raise


def test_save_creates_parsed_storage_dir_when_missing(tmp_path, monkeypatch):
    parsed_dir = tmp_path / "nested" / "parsed"
    monkeypatch.setattr(parsed_store.settings, "parsed_storage_dir", str(parsed_dir))
    result = MineruResult(blocks=[{"type": "text", "page_idx": 0, "text": "x"}], images={})

    parsed_store.save("doc_e", "e.pdf", result, tmp_path / "images")

    assert (parsed_dir / "doc_e.json").exists()
