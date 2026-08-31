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
