# MinerU 기반 PDF 파싱 도입 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pdf_ocr.py`의 PyMuPDF 레이아웃 분석을 MinerU HTTP 서비스 호출로 대체하고, PaddleOCR은 MinerU가 별도 이미지 블록으로만 남기는 도표/차트 내부 텍스트를 계속 보강한다. 기존 `PdfPage` 스키마, 청킹, RAG, 이미지 인용/서빙 파이프라인은 무변경으로 유지한다.

**Architecture:** 새 모듈 `app/services/mineru_client.py`가 `MineruClient`(HTTP 호출 계층)와 `build_pages()`(content_list.json 블록 → 페이지 단위 `PdfPage` 조립, 이미지 블록은 기존 `get_ocr_service()`로 OCR)를 제공한다. `document_processor.py`는 `extract_pages()` 호출을 `parse_and_build_pages()` 호출로 교체하고, 기존 PyPDFLoader 폴백 구조를 그대로 유지한다. `pdf_ocr.py`와 그 테스트는 마이그레이션 완료 후 삭제한다.

**Tech Stack:** FastAPI, httpx(이미 `requirements.txt`에 존재, 추가 설치 불필요), PyMuPDF(fitz, 테스트에서 최소 PDF 생성용으로만 남음), PaddleOCR(`ocr_service.py`, 무변경), pytest, `httpx.MockTransport`(HTTP 계층 테스트, 실제 서버 불필요).

**Spec:** [docs/superpowers/specs/2026-08-31-mineru-pdf-parsing-design.md](../specs/2026-08-31-mineru-pdf-parsing-design.md)

## Global Constraints

- 이 저장소는 `conftest.py`를 쓰지 않는다 — 각 테스트 파일이 필요한 스텁/헬퍼를 자체적으로 정의한다.
- 태스크당 커밋 1개. TDD: 실패하는 테스트 → 실패 확인 → 최소 구현 → 통과 확인 → 커밋.
- `backend/app/venv/bin/python -m pytest tests/ -v`가 매 태스크 후 전부 통과해야 한다.
- MinerU 파이썬 패키지(`mineru`/`magic_pdf`)는 **이 백엔드 venv에 설치하지 않는다** — MinerU는 별도 프로세스(HTTP 서비스)로만 존재한다. `requirements.txt`는 이 기능으로 인해 변경되지 않는다(`httpx==0.26.0`이 이미 있다).
- 이미지 인용 스킴은 그대로 유지한다: `image_id = md5(원본 바이트).hexdigest()[:16]`, 저장 경로 `{image_storage_dir}/{document_id}/{image_id}.png`, `documents.py`의 `_SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")` 화이트리스트와 호환되어야 한다(md5 hexdigest는 항상 이 정규식을 통과한다).
- OCR 라벨 마커는 `settings.ocr_block_prefix`(`"[이미지 텍스트]"`)를 그대로 재사용한다 — 새 마커를 만들지 않는다.
- `document_processor.py`, `chunking.py`, `rag_service.py`, `app/api/routes/documents.py`, `app/api/models/responses.py`는 이 계획에서 **document_processor.py의 import/load_pdf 두 곳 외에는 수정하지 않는다** — 스펙 2절의 "기존 하위 파이프라인 무변경" 원칙.
- 새 설정(`mineru_base_url`, `mineru_timeout`, `mineru_enabled`, `mineru_lang_list`)과 제거 대상 설정(`ocr_dpi`, `ocr_min_image_size`, `ocr_max_images_per_page`, `ocr_page_text_threshold`, `ocr_row_tolerance`, `ocr_layout_order`)은 스펙 3.6절 표와 정확히 일치해야 한다.
- **MinerU `/file_parse`의 실제 응답 형태(2026-08-31, 설치된 MinerU 3.4.5로 실측 확인, 스펙 3.2절)**: 요청 파일 필드명은 `files`(복수형, 단일 파일도 이 이름). `backend` 폼 필드는 기본값이 `hybrid-engine`(VLM 필요)이라 **반드시 `pipeline`을 명시**해야 한다. `return_content_list=true`/`return_images=true`도 기본값이 `false`라 명시적으로 켜야 한다. 응답은 `{"status", "error", "file_names": [...], "results": {<file_stem>: {"content_list": "<JSON 문자열>", "images": {img_path: "data:image/...;base64,..."}}}}` 형태의 작업 봉투다 — `content_list`는 블록 배열이 아니라 그 배열을 담은 **JSON 문자열**이라 한 번 더 파싱해야 하고, `images`는 파일 경로가 아니라 **base64 데이터 URI 딕셔너리**다. `/file_parse`는 동기 엔드포인트라 폴링은 불필요하다.

---

## Task 1: MinerU 서비스 기동 (운영 작업, 커밋 없음) — 완료, 실측값으로 갱신됨

이 태스크는 이 저장소에 코드를 추가하지 않는다. **2026-08-31에 실행 완료** —
아래는 실제로 사용된 명령과 실측 결과다(재현/재기동 시 그대로 사용 가능).

**Files:** 없음 (운영 절차)

- [x] **Step 1: MinerU를 별도 환경에 설치**

```bash
python3 -m venv ~/mineru-venv
~/mineru-venv/bin/pip install -U pip
~/mineru-venv/bin/pip install -U "mineru[core]"
```

설치됨: `mineru==3.4.5` (torch 2.13.0, CUDA 13 계열 의존성 포함). 백엔드 venv
`backend/app/venv`와 섞지 않았다 — Global Constraints 참고.

- [ ] **Step 2: API 서버 기동** (세션 종료 시 죽는 `nohup` 백그라운드 프로세스이므로, 새 세션에서 재실행 시 다시 필요)

```bash
nohup ~/mineru-venv/bin/mineru-api --host 0.0.0.0 --port 8100 > ~/mineru-api.log 2>&1 &
disown
sleep 5 && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8100/docs
```

Expected: `200`

- [x] **Step 3: 샘플 PDF로 `/file_parse` 응답 형태 확인 — 실측 완료**

실제 요청(필드명 `files`, 반드시 `backend=pipeline` 명시 — 기본값은
`hybrid-engine`이라 로컬 VLM을 요구한다):

```bash
curl -s -X POST http://127.0.0.1:8100/file_parse \
  -F "files=@/tmp/sample.pdf" \
  -F "backend=pipeline" \
  -F "lang_list=korean" \
  -F "return_content_list=true" \
  -F "return_images=true" \
  -F "return_md=false" \
  -o /tmp/mineru_response.json -w "HTTP %{http_code}\n" --max-time 180
```

**실측 결과**(Global Constraints에도 요약됨, 전체 근거는 스펙 3.2절):
응답은 `{"status": "completed", "error": null, "file_names": ["sample"],
"results": {"sample": {"content_list": "<JSON 문자열>", "images":
{<img_path>: "data:image/jpeg;base64,..."}}}}` 형태. `content_list`는
`json.loads()`가 한 번 더 필요한 문자열이고, `images`는 파일 경로가
아니라 base64 데이터 URI 딕셔너리다 — 최초 설계가 가정했던 "같은 호스트
파일시스템 공유"는 필요 없다. Task 2/3의 코드는 이 실측값을 반영해 이미
작성되어 있다.

- [x] **Step 4: `curl http://127.0.0.1:8100/docs`로 Swagger UI가 뜨는지 확인 — `200` 확인됨**

이후 태스크의 수동 스모크 테스트(Task 7)에서 이 서비스가 계속 떠 있어야
한다. 현재 이 세션에서 `nohup`으로 띄운 프로세스가 살아 있다(포트 8100).

---

## Task 2: `MineruClient` — MinerU 서비스 HTTP 호출 계층

**Files:**
- Create: `backend/app/services/mineru_client.py`
- Test: Create `backend/tests/test_mineru_client.py`

**Interfaces:**
- Produces: `MineruClient(base_url=None, timeout=None, lang_list=None, transport=None)`, `.parse_pdf(file_path: str) -> MineruResult`; `MineruResult(blocks: list[dict], images: dict[str, str])` (`images`는 `img_path -> "data:image/...;base64,..."` 딕셔너리) — 이후 태스크가 그대로 사용.

이 태스크의 요청/응답 형태는 Task 1에서 설치된 실제 MinerU 3.4.5 서비스로
실측 확인된 것이다(스펙 3.2절, Global Constraints) — 추측이 아니다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_mineru_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.mineru_client'` (설정값 `mineru_base_url`/`mineru_timeout`/`mineru_lang_list`도 아직 없으므로, 이를 먼저 추가해야 마지막 테스트가 의미 있게 실패/통과한다 — Step 3에서 `config.py`도 함께 수정).

- [ ] **Step 3: `config.py`에 설정 추가**

`backend/app/config.py`의 `# RAG Configuration` 앞에 추가:

```python
    # MinerU (PDF layout/table/formula parsing service)
    mineru_base_url: str = "http://127.0.0.1:8100"
    mineru_timeout: float = 300.0  # seconds; large PDFs take much longer than a single OCR call
    mineru_enabled: bool = True
    mineru_lang_list: list[str] = ["korean"]  # passed verbatim as /file_parse's lang_list form field
```

- [ ] **Step 4: `mineru_client.py` 구현**

```python
"""HTTP client for the self-hosted MinerU document-parsing service.

MinerU replaces this app's PyMuPDF-based PDF layout analysis - see
docs/superpowers/specs/2026-08-31-mineru-pdf-parsing-design.md. This module
only talks to the service and returns its content_list blocks plus their
images (as base64 data URIs - MinerU 3.4.5's /file_parse embeds images
inline rather than exposing a shared output directory, verified 2026-08-31).
Page-grouping and text assembly live in build_pages() (added in the next task).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from app.config import get_settings

settings = get_settings()


@dataclass
class MineruResult:
    """content_list blocks plus their images (img_path -> base64 data URI)."""

    blocks: list
    images: dict


class MineruClient:
    """Calls the MinerU HTTP service's /file_parse endpoint (synchronous - it
    waits for parsing to finish before responding, so no polling is needed).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        lang_list: Optional[list] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url or settings.mineru_base_url
        self.timeout = timeout if timeout is not None else settings.mineru_timeout
        self.lang_list = lang_list if lang_list is not None else settings.mineru_lang_list
        self._transport = transport

    def parse_pdf(self, file_path: str) -> MineruResult:
        """Upload a PDF and return its parsed content_list blocks and images.

        Raises httpx.HTTPStatusError on a non-2xx response, and RuntimeError
        when the response is 200 but the parsing task itself failed. Either
        way the caller decides whether to fall back (see
        document_processor.load_pdf).
        """
        with httpx.Client(
            base_url=self.base_url, timeout=self.timeout, transport=self._transport
        ) as client:
            with open(file_path, "rb") as f:
                response = client.post(
                    "/file_parse",
                    files={"files": (Path(file_path).name, f, "application/pdf")},
                    data={
                        "backend": "pipeline",  # server default is hybrid-engine, which needs a local VLM
                        "lang_list": self.lang_list,
                        "return_content_list": "true",
                        "return_images": "true",
                        "return_md": "false",
                    },
                )
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "completed" or payload.get("error"):
            raise RuntimeError(f"MinerU parsing failed: {payload.get('error') or payload.get('status')}")

        file_stem = payload["file_names"][0]
        entry = payload["results"][file_stem]
        return MineruResult(
            blocks=json.loads(entry["content_list"]),
            images=entry["images"],
        )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_mineru_client.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: (선택, 권장) 실제 서비스로 스모크 검증**

Task 1에서 띄운 실제 MinerU 서비스가 살아 있다면(`curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8100/docs`가 `200`), 실제 HTTP 호출로 한 번 더 검증한다:

```bash
cd backend && app/venv/bin/python -c "
from app.services.mineru_client import MineruClient
result = MineruClient().parse_pdf('/tmp/sample_img.pdf')
print(len(result.blocks), 'blocks')
print(list(result.images.keys()))
"
```

Expected: 예외 없이 블록 수와 이미지 키가 출력된다(샘플 PDF가 없으면 이 단계는 건너뛰고 Task 7의 스모크 테스트에서 검증한다).

- [ ] **Step 7: 커밋**

```bash
git add backend/app/services/mineru_client.py backend/tests/test_mineru_client.py backend/app/config.py
git commit -m "feat: add MineruClient HTTP layer for the self-hosted MinerU service"
```

---

## Task 3: `build_pages()` — content_list 블록을 `PdfPage`로 조립

**Files:**
- Modify: `backend/app/services/mineru_client.py`
- Modify: `backend/tests/test_mineru_client.py`

**Interfaces:**
- Consumes: `MineruResult`(Task 2, `blocks`/`images` 필드), `settings.ocr_block_prefix`(기존 `config.py`).
- Produces: `PdfPage(page_number, text, image_count=0, ocr_image_count=0, full_page_ocr=False, image_ids=[])`, `SupportsImageOcr` 프로토콜(`.image_to_text(image) -> str`), `build_pages(blocks: list[dict], images: dict, ocr: Optional[SupportsImageOcr], image_dir: Optional[Path] = None) -> list[PdfPage]` — Task 4와 `document_processor.py`가 그대로 사용.

`images`는 Task 2에서 확정된 실측 형태 그대로 `img_path -> "data:image/...;base64,..."` 딕셔너리다(파일 경로가 아니다).

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_mineru_client.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_mineru_client.py -v`
Expected: FAIL with `ImportError: cannot import name 'PdfPage'`

- [ ] **Step 3: `mineru_client.py`에 `build_pages()`와 지원 코드 추가**

`backend/app/services/mineru_client.py` 상단 import를 확장하고 파일 끝에 추가:

```python
# 파일 상단 import 블록을 아래로 교체
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
```

파일 끝(`MineruClient` 클래스 뒤)에 추가:

```python
class SupportsImageOcr(Protocol):
    """Minimal interface required from an OCR backend."""

    def image_to_text(self, image: Any) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class PdfPage:
    """One page assembled from MinerU's content_list blocks."""

    page_number: int  # 1-based
    text: str
    image_count: int = 0
    ocr_image_count: int = 0
    full_page_ocr: bool = False  # unused on the MinerU path; MinerU decides scanned-vs-not internally
    image_ids: list = field(default_factory=list)


def _format_ocr_block(text: str) -> str:
    """Wrap OCR output so retrieved chunks show where the text came from."""
    prefix = settings.ocr_block_prefix.strip()
    if not prefix:
        return text
    return f"{prefix}\n{text}"


def _save_image(image: Any, image_dir: Path, image_id: str) -> bool:
    """Persist a BGR numpy array as PNG. Never raises; returns False on failure."""
    try:
        from PIL import Image as PILImage

        image_dir.mkdir(parents=True, exist_ok=True)
        rgb = image[:, :, ::-1]  # BGR -> RGB
        PILImage.fromarray(rgb).save(image_dir / f"{image_id}.png", format="PNG")
        return True
    except Exception as e:
        logger.warning("Failed to save extracted image %s: %s", image_id, e)
        return False


def _decode_data_uri(data_uri: str) -> bytes:
    """Decode a 'data:image/...;base64,....' URI into raw image bytes."""
    import base64

    _, encoded = data_uri.split(",", 1)
    return base64.b64decode(encoded)


def _bgr_array_from_bytes(raw: bytes) -> Any:
    """Decode image bytes into a BGR numpy array (matches PaddleOCR's expected order)."""
    import io

    import numpy as np
    from PIL import Image as PILImage

    rgb = np.array(PILImage.open(io.BytesIO(raw)).convert("RGB"))
    return rgb[:, :, ::-1]


def _text_of(block: dict) -> str:
    if block.get("type") == "table":
        return (block.get("table_body") or "").strip()
    return (block.get("text") or "").strip()


def _ocr_image_block(
    block: dict,
    images: dict,
    ocr: SupportsImageOcr,
    cache: dict,
    image_dir: Optional[Path],
) -> tuple:
    """OCR one image block, reusing results for repeated image bytes.

    The image is saved whenever image_dir is given, even if OCR finds no
    text - only the citation (image_id surfaced to the caller) is gated on
    recognised text, matching the file-saving contract PaddleOCR splicing
    already used for embedded PDF images.
    """
    raw = _decode_data_uri(images[block["img_path"]])
    key = hashlib.md5(raw).hexdigest()[:16]

    if key in cache:
        return cache[key]

    image = _bgr_array_from_bytes(raw)
    text = (ocr.image_to_text(image) or "").strip()

    image_id = None
    if image_dir is not None and _save_image(image, image_dir, key):
        image_id = key

    result = (text, image_id)
    cache[key] = result
    return result


def build_pages(
    blocks: list,
    images: dict,
    ocr: Optional[SupportsImageOcr],
    image_dir: Optional[Path] = None,
) -> list:
    """Group content_list blocks by page and assemble each page's PdfPage.

    Blocks arrive in MinerU's reading order already - this only groups by
    page_idx, it does not re-sort within a page. images maps each image
    block's img_path to a base64 data URI (MinerU's /file_parse response
    shape - see MineruResult). ocr=None (settings.ocr_enabled is False)
    skips OCR augmentation of image blocks entirely: they contribute no
    text and no citation.
    """
    by_page: dict = {}
    for block in blocks:
        by_page.setdefault(block["page_idx"], []).append(block)

    cache: dict = {}
    pages = []
    for page_idx in sorted(by_page):
        page_blocks = by_page[page_idx]
        parts = []
        ocr_image_count = 0
        image_ids = []

        for block in page_blocks:
            if block.get("type") == "image":
                if ocr is None:
                    continue
                text, image_id = _ocr_image_block(block, images, ocr, cache, image_dir)
                if not text:
                    continue
                ocr_image_count += 1
                if image_id is not None:
                    image_ids.append(image_id)
                parts.append(_format_ocr_block(text))
            else:
                text = _text_of(block)
                if text:
                    parts.append(text)

        pages.append(
            PdfPage(
                page_number=page_idx + 1,
                text="\n\n".join(parts),
                image_count=sum(1 for b in page_blocks if b.get("type") == "image"),
                ocr_image_count=ocr_image_count,
                image_ids=image_ids,
            )
        )

    return pages
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_mineru_client.py -v`
Expected: PASS (전체, Task 2의 4개 + 이번 태스크의 8개)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/mineru_client.py backend/tests/test_mineru_client.py
git commit -m "feat: assemble MinerU content_list blocks into PdfPage objects"
```

---

## Task 4: `parse_and_build_pages()` — 클라이언트 호출과 페이지 조립을 결합

**Files:**
- Modify: `backend/app/services/mineru_client.py`
- Modify: `backend/tests/test_mineru_client.py`

**Interfaces:**
- Consumes: `MineruClient`, `MineruResult`(Task 2, `blocks`/`images`), `build_pages`, `PdfPage`, `SupportsImageOcr`(Task 3).
- Produces: `parse_and_build_pages(file_path: str, ocr: Optional[SupportsImageOcr], image_dir: Optional[Path] = None, client: Optional[MineruClient] = None) -> list[PdfPage]` — Task 5의 `document_processor.load_pdf()`가 그대로 사용.

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_mineru_client.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_mineru_client.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_and_build_pages'`

- [ ] **Step 3: `parse_and_build_pages()` 구현**

`backend/app/services/mineru_client.py` 끝에 추가:

```python
def parse_and_build_pages(
    file_path: str,
    ocr: Optional[SupportsImageOcr],
    image_dir: Optional[Path] = None,
    client: Optional[MineruClient] = None,
) -> list:
    """Parse a PDF via MinerU and assemble it into PdfPage objects.

    ocr=None skips OCR augmentation of image blocks entirely (see
    build_pages) - the caller (document_processor.load_pdf) decides this
    based on settings.ocr_enabled.
    """
    if client is None:
        client = MineruClient()

    result = client.parse_pdf(file_path)
    return build_pages(result.blocks, result.images, ocr, image_dir)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_mineru_client.py -v`
Expected: PASS (전체)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/mineru_client.py backend/tests/test_mineru_client.py
git commit -m "feat: add parse_and_build_pages combinator for the MinerU PDF path"
```

---

## Task 5: `document_processor.py`를 MinerU 경로로 전환

**Files:**
- Modify: `backend/app/services/document_processor.py`
- Modify: `backend/tests/test_document_processor_ocr.py`

**Interfaces:**
- Consumes: `parse_and_build_pages`, `PdfPage`(Task 2-4).
- Produces: `DocumentProcessor(ocr=None, mineru_client=None)` — `mineru_client`는 새 생성자 인자(주입 가능, 기본값 `None`은 `parse_and_build_pages`가 실제 `MineruClient()`를 만들게 둔다는 뜻).

- [ ] **Step 1: `document_processor.py` 수정**

import 교체:

```python
# 기존
from app.services.pdf_ocr import PdfPage, extract_pages
# 신규
from app.services.mineru_client import PdfPage, parse_and_build_pages
```

`__init__`과 `load_pdf` 교체:

```python
    def __init__(self, ocr=None, mineru_client=None):
        # Defaults to the shared PaddleOCR service; injectable for tests.
        self.ocr = ocr
        # Defaults to a real MineruClient(); injectable for tests.
        self.mineru_client = mineru_client

    def load_pdf(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Load a PDF via MinerU, augmenting image blocks with PaddleOCR text.

        Falls back to plain text extraction if MinerU is disabled or
        unavailable, so an upload never fails just because the parsing
        service could not run.

        Args:
            file_path: Path to the PDF
            filename: Original filename
            document_id: Document ID; required to persist extracted images.
                Without it, text extraction proceeds unchanged and no images
                are saved.

        Returns:
            List of Document objects, one per page
        """
        if not settings.mineru_enabled:
            return PyPDFLoader(file_path).load()

        image_dir = Path(settings.image_storage_dir) / document_id if document_id else None

        ocr = None
        if settings.ocr_enabled:
            ocr = self.ocr
            if ocr is None:
                from app.services.ocr_service import get_ocr_service

                ocr = get_ocr_service()

        try:
            pages: list[PdfPage] = parse_and_build_pages(
                file_path, ocr=ocr, image_dir=image_dir, client=self.mineru_client
            )
        except Exception as e:
            logger.warning(
                f"MinerU extraction failed for {filename} ({e}); "
                "falling back to text-only extraction"
            )
            return PyPDFLoader(file_path).load()

        documents = []
        for page in pages:
            if not page.text.strip():
                continue
            documents.append(
                Document(
                    page_content=page.text,
                    metadata={
                        "page": page.page_number - 1,  # 0-based, as PyPDFLoader
                        "page_number": page.page_number,
                        "ocr_used": page.ocr_image_count > 0,
                        "ocr_image_count": page.ocr_image_count,
                        "full_page_ocr": page.full_page_ocr,
                        # Chroma metadata values must be str/int/float/bool, so a
                        # list can't be stored directly; image ids never contain
                        # commas (md5 hexdigest), so joining is lossless.
                        "image_ids": ",".join(page.image_ids),
                    },
                )
            )

        return documents
```

(다른 메서드 `load_document`/`chunk_documents`/`process_file`는 무변경.)

- [ ] **Step 2: 기존 이미지 관련 테스트를 FakeMineruClient 기반으로 재작성**

`backend/tests/test_document_processor_ocr.py`를 열어 상단의 `StubOCR`/`_pdf_with_image` 정의와, 이를 사용하는 모든 테스트를 아래로 **전체 교체**한다(파일 하단의 청킹 전략 테스트들, `class _FakeSplitter` 이하는 그대로 둔다):

```python
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

    monkeypatch.setattr(module.settings, "image_storage_dir", str(tmp_path / "images_out"))
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
```

`class _FakeSplitter` 정의는 그대로 둔다. 그 바로 아래 `test_semantic_strategy_does_not_write_chunk_size_or_overlap_metadata`는 여전히 `_pdf_with_image`/기본 `StubOCR(...)` 단독 생성을 쓰고 있으므로 이 두 곳만 고친다 — `DocumentProcessor(...)` 생성에 `mineru_client=` 인자를 추가:

```python
    processor = DocumentProcessor(
        ocr=StubOCR("그림 안의 설명 문장"), mineru_client=_client_with_text_and_image(tmp_path)
    )
```

그리고 그 테스트의 `processor.process_file(_pdf_with_image(tmp_path), "diagram.pdf", chunking_strategy="semantic")` 호출도 `processor.process_file("ignored.pdf", "diagram.pdf", chunking_strategy="semantic")`로 바꾼다.

- [ ] **Step 3: 테스트 실행**

Run: `cd backend && app/venv/bin/python -m pytest tests/test_document_processor_ocr.py -v`
Expected: PASS (전체)

- [ ] **Step 4: 전체 스위트 실행 (pdf_ocr.py는 아직 삭제 전이므로 test_pdf_ocr.py도 통과해야 한다)**

Run: `cd backend && app/venv/bin/python -m pytest tests/ -v`
Expected: PASS (전체 - `test_pdf_ocr.py`는 아직 pdf_ocr.py를 직접 테스트하고 있으므로 그대로 통과해야 정상)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/document_processor.py backend/tests/test_document_processor_ocr.py
git commit -m "feat: route DocumentProcessor.load_pdf through MinerU instead of PyMuPDF"
```

---

## Task 6: `pdf_ocr.py` 및 그 테스트 삭제, 잔여 참조 정리

**Files:**
- Delete: `backend/app/services/pdf_ocr.py`
- Delete: `backend/tests/test_pdf_ocr.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/services/rag_service.py:20` (주석 한 줄)

- [ ] **Step 1: 잔여 참조 확인**

Run: `cd backend && grep -rln "pdf_ocr\|extract_pages" app/ tests/ | grep -v venv`
Expected 출력: `app/services/rag_service.py`뿐이어야 한다(주석 한 줄, Step 4에서 수정). 다른 파일이 나오면 그 파일을 마이그레이션에서 놓친 것이므로 삭제 전에 먼저 처리한다.

- [ ] **Step 2: 파일 삭제**

```bash
git rm backend/app/services/pdf_ocr.py backend/tests/test_pdf_ocr.py
```

- [ ] **Step 3: `config.py`에서 PyMuPDF 레이아웃 전용 설정 제거**

`backend/app/config.py`에서 다음 6개 줄을 삭제한다(주석 포함):

```python
    ocr_dpi: int = 200  # Render resolution for image regions
    ocr_min_image_size: float = 40.0  # Skip icons/rules smaller than this (pt)
    ocr_max_images_per_page: int = 20  # Above this, OCR the whole page once
    ocr_page_text_threshold: int = 30  # Fewer native chars => treat as scanned
    ocr_row_tolerance: float = 10.0  # Blocks within this gap (pt) share a row
    ocr_layout_order: str = "position"  # "position" or "native" block order
```

`ocr_enabled` 줄의 주석을 스펙 3.6절에 맞게 갱신한다:

```python
    # 기존
    ocr_enabled: bool = True
    # 신규
    ocr_enabled: bool = True  # Whether image blocks get PaddleOCR augmentation (MinerU parsing itself is controlled by mineru_enabled)
```

- [ ] **Step 4: `rag_service.py`의 주석 갱신**

`backend/app/services/rag_service.py:20` 부근:

```python
# 기존
# 본문에 이미 인라인으로 박혀 있으므로(pdf_ocr._format_ocr_block), 여기서는 LLM에게
# 신규
# 본문에 이미 인라인으로 박혀 있으므로(mineru_client._format_ocr_block), 여기서는 LLM에게
```

- [ ] **Step 5: 전체 테스트 실행**

Run: `cd backend && app/venv/bin/python -m pytest tests/ -v`
Expected: PASS (전체 — `test_pdf_ocr.py`가 삭제되었으므로 수집 대상에서 사라진다)

- [ ] **Step 6: 커밋**

```bash
git add -A backend/app/services/pdf_ocr.py backend/tests/test_pdf_ocr.py backend/app/config.py backend/app/services/rag_service.py
git commit -m "chore: remove PyMuPDF layout pipeline, superseded by MinerU"
```

---

## Task 7: 수동 엔드투엔드 스모크 테스트

이 저장소는 라우트 레벨 `TestClient` 테스트가 없는 관례를 따른다(기존 청킹 전략 기능 도입 시와 동일) — 실제 서비스 연동 확인은 여기서 수동으로 한다. 이 태스크는 코드 변경이 없으므로 커밋도 없다.

**Files:** 없음

- [ ] **Step 1: MinerU 서비스가 떠 있는지 확인**

Run: `curl -s http://127.0.0.1:8100/docs -o /dev/null -w "%{http_code}\n"`
Expected: `200`

- [ ] **Step 2: 백엔드/프런트엔드 기동**

`/run` 스킬로 기동하거나:

```bash
cd backend && app/venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
cd frontend && npm run dev &
```

- [ ] **Step 3: 표/수식이 포함된 PDF 업로드**

브라우저에서 표가 포함된 PDF를 업로드하고, 업로드 응답에 `chunking_strategy`가 정상 포함되는지, 처리 시간이 과도하게 길지 않은지 확인한다.

- [ ] **Step 4: 채팅에서 표 내용 검색**

표 안의 특정 값(예: 특정 숫자나 항목명)으로 질문해 그 청크가 검색되는지 확인한다 — 기존에는 표가 OCR로 뒤섞여 검색 품질이 낮았던 부분이다.

- [ ] **Step 5: 도표/차트가 포함된 PDF로 이미지 인용 확인**

도표가 포함된 PDF를 업로드하고, 그 도표 내용과 관련된 질문을 던져 출처 카드에 원본 이미지 썸네일이 여전히 뜨는지 확인한다(기존 이미지 인용 기능이 MinerU 경로에서도 동작하는지의 최종 확인).

- [ ] **Step 6: MinerU 서비스를 잠시 내리고 폴백 확인**

MinerU 서비스 프로세스를 중지한 뒤 PDF를 업로드해 업로드가 실패하지 않고(텍스트 전용으로) 성공하는지 확인한다. 확인 후 서비스를 다시 올린다.

- [ ] **Step 7: 문서 삭제 시 이미지 정리 확인**

업로드했던 문서를 삭제하고 `app/storage/images/{document_id}/` 디렉터리가 제거되는지 확인한다(기존 기능, 회귀 여부만 확인).

---

## 검증 요약

- 매 태스크 후: `cd backend && app/venv/bin/python -m pytest tests/ -v` 전체 통과
- Task 6 이후: `grep -rln "pdf_ocr" backend/app backend/tests`가 빈 결과여야 한다
- Task 7(수동): 표/수식 검색 품질, 이미지 인용 유지, MinerU 장애 시 폴백, 문서 삭제 시 이미지 정리 — 4가지 모두 육안 확인
