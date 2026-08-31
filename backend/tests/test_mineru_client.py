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


from app.services.mineru_client import parse_and_build_pages


class FakeMineruClient:
    """Stands in for MineruClient: returns canned blocks instead of calling HTTP."""

    def __init__(self, blocks, images):
        self.blocks = blocks
        self.images = images

    def parse_pdf(self, file_path):
        return MineruResult(blocks=self.blocks, images=self.images)


def test_parse_and_build_pages_combines_client_and_page_assembly():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [
        {"type": "text", "page_idx": 0, "text": "본문"},
        {"type": "image", "page_idx": 0, "img_path": "images/img1.png"},
    ]
    client = FakeMineruClient(blocks, images)

    pages = parse_and_build_pages("ignored.pdf", ocr=StubOCR("도표 텍스트"), client=client)

    assert len(pages) == 1
    assert "본문" in pages[0].text
    assert "도표 텍스트" in pages[0].text


def test_ocr_none_is_forwarded_to_build_pages():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [{"type": "image", "page_idx": 0, "img_path": "images/img1.png"}]
    client = FakeMineruClient(blocks, images)

    pages = parse_and_build_pages("ignored.pdf", ocr=None, client=client)

    assert pages[0].text == ""


def test_default_client_is_a_real_mineru_client(monkeypatch):
    """client=None을 넘기면 실제 MineruClient()를 만든다."""
    created = {}

    class RecordingClient:
        def __init__(self):
            created["called"] = True

        def parse_pdf(self, file_path):
            return MineruResult(blocks=[], images={})

    monkeypatch.setattr("app.services.mineru_client.MineruClient", RecordingClient)

    parse_and_build_pages("ignored.pdf", ocr=None, client=None)

    assert created.get("called") is True


def test_image_block_resolves_img_path_with_a_directory_prefix_against_a_bare_filename_key():
    """Reproduces a real defect found via manual smoke testing against the
    live MinerU 3.4.5 service (2026-08-31): content_list blocks' img_path
    has an "images/" prefix, but the images dict is keyed by the bare
    filename - not the same string. Every other fixture in this file
    accidentally uses matching keys on both sides and would NOT catch this."""
    images = {"onlyfilename.png": _png_data_uri()}
    blocks = [{"type": "image", "page_idx": 0, "img_path": "images/onlyfilename.png"}]

    pages = build_pages(blocks, images=images, ocr=StubOCR("도표 텍스트"))

    assert "도표 텍스트" in pages[0].text


def test_a_non_image_typed_block_with_img_path_is_still_ocr_ed_and_cited(tmp_path):
    """Reproduces a real MinerU response observed live (2026-08-31): a figure
    block came back as type="chart", not "image", but with the same
    img_path/empty-text shape. build_pages() must not silently drop it."""
    images = {"images/chart1.png": _png_data_uri()}
    blocks = [
        {"type": "text", "page_idx": 0, "text": "본문 앞부분"},
        {"type": "chart", "page_idx": 0, "img_path": "images/chart1.png", "text": ""},
    ]
    image_dir = tmp_path / "saved"

    pages = build_pages(blocks, images=images, ocr=StubOCR("차트 안의 텍스트"), image_dir=image_dir)

    page = pages[0]
    assert "[이미지 텍스트]\n차트 안의 텍스트" in page.text
    assert page.ocr_image_count == 1
    assert len(page.image_ids) == 1
    assert (image_dir / f"{page.image_ids[0]}.png").exists()


def test_table_block_with_an_img_path_screenshot_still_uses_table_body_not_ocr():
    """Per the spec's schema, table blocks also carry img_path (a
    screenshot) - this must NOT be routed through the image-OCR path even
    though it now matches "has img_path"."""
    blocks = [
        {
            "type": "table",
            "page_idx": 0,
            "img_path": "images/table_screenshot.jpg",
            "table_body": "<table><tr><td>매출</td><td>120억</td></tr></table>",
        }
    ]
    ocr = StubOCR("이건 호출되면 안 됨")

    pages = build_pages(blocks, images={}, ocr=ocr)

    assert "<table><tr><td>매출</td><td>120억</td></tr></table>" in pages[0].text
    assert ocr.calls == []


def test_a_block_that_raises_does_not_abort_the_rest_of_the_page():
    """A missing images-dict entry (or any other per-block failure) must
    cost only that block, not the whole page - pdf_ocr.py (this module's
    predecessor) guaranteed this per-image; build_pages() must too."""
    blocks = [
        {"type": "text", "page_idx": 0, "text": "본문 앞부분"},
        {"type": "image", "page_idx": 0, "img_path": "images/missing.png"},  # not in images dict
        {"type": "text", "page_idx": 0, "text": "본문 뒷부분"},
    ]

    pages = build_pages(blocks, images={}, ocr=StubOCR("무시됨"))

    assert "본문 앞부분" in pages[0].text
    assert "본문 뒷부분" in pages[0].text
    assert pages[0].ocr_image_count == 0


def test_a_block_missing_page_idx_defaults_to_page_one_instead_of_crashing():
    blocks = [{"type": "text", "text": "page_idx 없는 블록"}]  # no page_idx key at all

    pages = build_pages(blocks, images={}, ocr=StubOCR())

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "page_idx 없는 블록" in pages[0].text


def test_table_block_with_empty_table_body_falls_back_to_ocr():
    images = {"images/table_shot.png": _png_data_uri()}
    blocks = [
        {
            "type": "table",
            "page_idx": 0,
            "img_path": "images/table_shot.png",
            "table_body": "",  # recognition failed server-side
        }
    ]

    pages = build_pages(blocks, images=images, ocr=StubOCR("표 이미지에서 읽은 텍스트"))

    assert "[이미지 텍스트]\n표 이미지에서 읽은 텍스트" in pages[0].text
    assert pages[0].ocr_image_count == 1


def test_equation_block_with_empty_text_falls_back_to_ocr():
    images = {"images/eqn_shot.png": _png_data_uri()}
    blocks = [
        {
            "type": "equation",
            "page_idx": 0,
            "img_path": "images/eqn_shot.png",
            "text": "",  # LaTeX recognition failed
        }
    ]

    pages = build_pages(blocks, images=images, ocr=StubOCR("수식 이미지에서 읽은 텍스트"))

    assert "[이미지 텍스트]\n수식 이미지에서 읽은 텍스트" in pages[0].text
    assert pages[0].ocr_image_count == 1


from app.services.mineru_client import persist_block_image


def test_persist_block_image_saves_and_returns_content_addressed_id(tmp_path):
    import hashlib

    data_uri = _png_data_uri()
    raw = _decode_data_uri_for_test(data_uri)
    expected_id = hashlib.md5(raw).hexdigest()[:16]
    block = {"type": "image", "page_idx": 0, "img_path": "images/fig1.png"}
    images = {"images/fig1.png": data_uri}
    image_dir = tmp_path / "saved"

    result = persist_block_image(block, images, image_dir)

    assert result == expected_id
    assert (image_dir / f"{expected_id}.png").exists()


def _decode_data_uri_for_test(data_uri: str) -> bytes:
    import base64

    _, encoded = data_uri.split(",", 1)
    return base64.b64decode(encoded)


def test_persist_block_image_returns_none_when_no_img_path(tmp_path):
    block = {"type": "text", "page_idx": 0, "text": "본문"}

    result = persist_block_image(block, images={}, image_dir=tmp_path / "saved")

    assert result is None


def test_persist_block_image_returns_none_when_images_dict_missing_key(tmp_path):
    block = {"type": "image", "page_idx": 0, "img_path": "images/missing.png"}

    result = persist_block_image(block, images={}, image_dir=tmp_path / "saved")

    assert result is None


def test_persist_block_image_returns_none_when_image_data_is_corrupted(tmp_path):
    import base64

    block = {"type": "image", "page_idx": 0, "img_path": "images/corrupt.png"}
    corrupted_uri = "data:image/png;base64," + base64.b64encode(b"not a real png").decode()
    images = {"images/corrupt.png": corrupted_uri}

    result = persist_block_image(block, images, tmp_path / "saved")

    assert result is None


def test_on_parsed_callback_receives_raw_result_before_build_pages_consumes_it():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [{"type": "text", "page_idx": 0, "text": "본문"}]
    client = FakeMineruClient(blocks, images)
    received = []

    parse_and_build_pages(
        "ignored.pdf", ocr=None, client=client, on_parsed=lambda result: received.append(result)
    )

    assert len(received) == 1
    assert received[0].blocks == blocks
    assert received[0].images == images


def test_on_parsed_defaults_to_none_and_is_optional():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [{"type": "text", "page_idx": 0, "text": "본문"}]
    client = FakeMineruClient(blocks, images)

    pages = parse_and_build_pages("ignored.pdf", ocr=None, client=client)

    assert pages[0].text == "본문"


def test_on_parsed_callback_exception_does_not_break_parsing():
    images = {"images/img1.png": _png_data_uri()}
    blocks = [{"type": "text", "page_idx": 0, "text": "본문"}]
    client = FakeMineruClient(blocks, images)

    def failing_callback(result):
        raise ValueError("callback error")

    pages = parse_and_build_pages(
        "ignored.pdf", ocr=None, client=client, on_parsed=failing_callback
    )

    assert len(pages) == 1
    assert pages[0].text == "본문"
