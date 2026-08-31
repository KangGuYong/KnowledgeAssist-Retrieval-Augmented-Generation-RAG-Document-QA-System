"""MineruClient talks to the self-hosted MinerU HTTP service and returns
content_list.json blocks plus their referenced images as base64 data URIs.
"""

import json

import httpx
import pytest

from app.services.mineru_client import MineruClient, MineruResult


def _task_envelope(blocks, images=None, file_stem="sample", status="completed", error=None):
    return {
        "status": status,
        "error": error,
        "file_names": [file_stem],
        "results": {
            file_stem: {
                "content_list": json.dumps(blocks),
                "images": images or {},
            }
        },
    }


def test_parse_pdf_posts_the_file_with_pipeline_backend_and_returns_blocks(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    blocks = [{"type": "text", "page_idx": 0, "text": "hello"}]
    images = {"images/img1.jpg": "data:image/jpeg;base64,Zm9v"}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/file_parse"
        # multipart field name is "files" (plural) - MinerU 3.4.5's actual schema
        assert b'name="files"' in request.content
        assert b'name="backend"' in request.content
        assert b"pipeline" in request.content
        return httpx.Response(200, json=_task_envelope(blocks, images))

    transport = httpx.MockTransport(handle)
    client = MineruClient(base_url="http://mineru.local", transport=transport)

    result = client.parse_pdf(str(pdf_path))

    assert isinstance(result, MineruResult)
    assert result.blocks == blocks
    assert result.images == images


def test_parse_pdf_raises_on_http_error(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = MineruClient(base_url="http://mineru.local", transport=httpx.MockTransport(handle))

    with pytest.raises(httpx.HTTPStatusError):
        client.parse_pdf(str(pdf_path))


def test_parse_pdf_raises_when_task_status_is_not_completed(tmp_path):
    """HTTP 200이어도 status/error가 실패를 나타낼 수 있다 (실측: 지원하지 않는
    파일 형식은 400을 주지만, 파싱 자체 실패는 200 + status="failed"로 올 수 있다)."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_task_envelope([], status="failed", error="parsing crashed")
        )

    client = MineruClient(base_url="http://mineru.local", transport=httpx.MockTransport(handle))

    with pytest.raises(RuntimeError, match="parsing crashed"):
        client.parse_pdf(str(pdf_path))


def test_base_url_timeout_and_lang_list_default_to_settings():
    client = MineruClient()

    from app.config import get_settings

    settings = get_settings()
    assert client.base_url == settings.mineru_base_url
    assert client.timeout == settings.mineru_timeout
    assert client.lang_list == settings.mineru_lang_list


from app.services.mineru_client import PdfPage, build_pages


class StubOCR:
    """Stands in for PaddleOCR: records what it was asked to read."""

    def __init__(self, text="인식된 텍스트"):
        self.text = text
        self.calls = []

    def image_to_text(self, image):
        self.calls.append(image)
        return self.text


def _png_data_uri(color=(10, 90, 200), size=(20, 15)):
    """MinerU's /file_parse images dict value shape: a base64 data URI."""
    import base64
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_text_table_equation_blocks_are_joined_in_reading_order():
    blocks = [
        {"type": "text", "page_idx": 0, "text": "제목입니다"},
        {"type": "table", "page_idx": 0, "table_body": "<table><tr><td>120억</td></tr></table>"},
        {"type": "equation", "page_idx": 0, "text": "E = mc^2"},
    ]

    pages = build_pages(blocks, images={}, ocr=StubOCR())

    assert len(pages) == 1
    page = pages[0]
    assert isinstance(page, PdfPage)
    assert page.page_number == 1
    assert "제목입니다" in page.text
    assert "<table><tr><td>120억</td></tr></table>" in page.text
    assert "E = mc^2" in page.text
    assert page.text.index("제목입니다") < page.text.index("<table>") < page.text.index("E = mc^2")


def test_pages_are_grouped_by_page_idx_zero_based_to_one_based():
    blocks = [
        {"type": "text", "page_idx": 0, "text": "첫 페이지"},
        {"type": "text", "page_idx": 1, "text": "둘째 페이지"},
    ]

    pages = build_pages(blocks, images={}, ocr=StubOCR())

    assert [p.page_number for p in pages] == [1, 2]
    assert pages[0].text == "첫 페이지"
    assert pages[1].text == "둘째 페이지"


def test_image_block_is_ocr_ed_and_labelled(tmp_path):
    images = {"images/img1.png": _png_data_uri()}
    blocks = [
        {"type": "text", "page_idx": 0, "text": "그림 설명 앞"},
        {"type": "image", "page_idx": 0, "img_path": "images/img1.png"},
    ]
    image_dir = tmp_path / "saved"

    pages = build_pages(blocks, images=images, ocr=StubOCR("표: 매출 120억"), image_dir=image_dir)

    page = pages[0]
    assert "[이미지 텍스트]\n표: 매출 120억" in page.text
    assert page.ocr_image_count == 1
    assert len(page.image_ids) == 1
    assert (image_dir / f"{page.image_ids[0]}.png").exists()


def test_identical_image_bytes_across_pages_are_ocr_ed_once(tmp_path):
    same_uri = _png_data_uri(color=(5, 5, 5))
    images = {
        "images/logo_p1.png": same_uri,
        "images/logo_p2.png": same_uri,  # same bytes, different img_path (repeated across pages)
    }
    blocks = [
        {"type": "image", "page_idx": 0, "img_path": "images/logo_p1.png"},
        {"type": "image", "page_idx": 1, "img_path": "images/logo_p2.png"},
    ]
    ocr = StubOCR("로고 텍스트")

    pages = build_pages(blocks, images=images, ocr=ocr, image_dir=tmp_path / "saved")

    assert len(ocr.calls) == 1
    assert pages[0].image_ids == pages[1].image_ids == [pages[0].image_ids[0]]


def test_image_with_no_recognised_text_contributes_nothing():
    images = {"images/blank.png": _png_data_uri()}
    blocks = [{"type": "image", "page_idx": 0, "img_path": "images/blank.png"}]

    pages = build_pages(blocks, images=images, ocr=StubOCR(""))

    assert len(pages) == 1
    assert pages[0].text == ""
    assert pages[0].image_ids == []


def test_ocr_none_skips_image_blocks_without_erroring():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [
        {"type": "text", "page_idx": 0, "text": "본문"},
        {"type": "image", "page_idx": 0, "img_path": "images/img1.png"},
    ]

    pages = build_pages(blocks, images=images, ocr=None)

    assert pages[0].text == "본문"
    assert pages[0].image_ids == []


def test_image_save_failure_does_not_break_text_extraction(tmp_path, monkeypatch):
    images = {"images/img1.png": _png_data_uri()}
    blocks = [{"type": "image", "page_idx": 0, "img_path": "images/img1.png"}]

    monkeypatch.setattr("app.services.mineru_client._save_image", lambda *a, **k: False)

    pages = build_pages(
        blocks, images=images, ocr=StubOCR("인식됨"), image_dir=tmp_path / "saved"
    )

    assert "인식됨" in pages[0].text
    assert pages[0].image_ids == []


def test_no_image_dir_skips_saving_without_failing():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [{"type": "image", "page_idx": 0, "img_path": "images/img1.png"}]

    pages = build_pages(blocks, images=images, ocr=StubOCR("인식됨"))

    assert "인식됨" in pages[0].text
    assert pages[0].image_ids == []
