# MinerU 파싱 결과 뷰어 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MinerU가 파싱한 원본 블록(text/title/table/equation/image)을 문서별로 저장하고, 이를 조회하는 API 2개와 프론트엔드 뷰어 탭을 추가해 사용자가 파싱 결과를 눈으로 확인할 수 있게 한다.

**Architecture:** `MineruClient.parse_pdf()`가 반환한 원본 블록을 `build_pages()`가 소비하기 전에 새 콜백(`on_parsed`)으로 가로채, 문서별 JSON 파일(`app/storage/parsed/{document_id}.json`)로 저장한다. 이 저장은 기존 청킹/임베딩 파이프라인과 완전히 분리된 읽기 전용 부가 기능이며 절대 예외를 전파하지 않는다. 새 API 2개가 이 JSON을 읽어 목록/상세를 반환하고, 프론트엔드는 기존 단일 페이지에 탭을 하나 추가해 이를 렌더링한다.

**Tech Stack:** FastAPI, Pydantic, pytest (백엔드) / React, TypeScript, axios, DOMPurify(신규) (프론트엔드)

**Spec:** [docs/superpowers/specs/2026-08-31-mineru-result-viewer-design.md](../specs/2026-08-31-mineru-result-viewer-design.md)

## Global Constraints

- 원본 그대로 노출한다 — LaTeX 렌더링(KaTeX 등), 마크다운 변환 등 가공은 하지 않는다(스펙 3.4절).
- 저장 실패는 절대 업로드/파싱 파이프라인을 막지 않는다 — `parsed_store.save()`는 예외를 전파하지 않는다(스펙 2절).
- 새 DB를 들이지 않는다 — 파일시스템 JSON 저장(스펙 2절).
- 표 HTML을 `dangerouslySetInnerHTML`로 렌더링하기 전 반드시 DOMPurify로 sanitize한다(스펙 3.4절, XSS 방지).
- `react-router` 등 라우팅 라이브러리를 추가하지 않는다 — 기존처럼 `App.tsx`의 로컬 state 탭 토글로 화면을 전환한다(스펙 3.4절).
- 기존 `GET /api/v1/documents/`(플레이스홀더)와 `DELETE /api/v1/documents/{document_id}`는 건드리지 않는다(스펙 3.3절).

---

## Task 1: `mineru_client.py` — 블록 이미지 저장 헬퍼 + `on_parsed` 콜백

**Files:**
- Modify: `backend/app/services/mineru_client.py`
- Test: `backend/tests/test_mineru_client.py`

**Interfaces:**
- Produces:
  - `persist_block_image(block: dict, images: dict, image_dir: Path) -> Optional[str]` — 블록에 `img_path`가 있으면 이미지를 디코드해 `image_dir`에 PNG로 저장하고 content-addressed `image_id`(md5 hexdigest[:16])를 반환한다. `img_path`가 없거나 `images` 딕셔너리에서 찾을 수 없으면 `None`.
  - `parse_and_build_pages(file_path, ocr, image_dir=None, client=None, on_parsed=None) -> list[PdfPage]` — 새 키워드 인자 `on_parsed: Optional[Callable[[MineruResult], None]]`. 주어지면 `client.parse_pdf()` 직후, `build_pages()`가 블록을 소비하기 전에 원본 `MineruResult`로 호출된다.

- [ ] **Step 1: 실패하는 테스트 작성 — `persist_block_image`**

`backend/tests/test_mineru_client.py` 맨 끝(407번째 줄 이후)에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && python -m pytest tests/test_mineru_client.py -k persist_block_image -v`
Expected: FAIL with `ImportError: cannot import name 'persist_block_image'`

- [ ] **Step 3: `persist_block_image` 구현**

`backend/app/services/mineru_client.py`의 `from typing import Any, Optional, Protocol`(16번째 줄)을 다음으로 교체:

```python
from typing import Any, Callable, Optional, Protocol
```

`_bgr_array_from_bytes` 정의(152-160번째 줄) 직후, `_text_of` 정의(163번째 줄) 앞에 추가:

```python
def persist_block_image(block: dict, images: dict, image_dir: Path) -> Optional[str]:
    """블록의 img_path 이미지를 image_dir에 저장한다. build_pages()의 OCR
    필요 여부(_needs_ocr) 판단과 무관하게, img_path가 있는 모든 블록에
    무조건 적용된다 - 파싱 결과 뷰어는 OCR 대상 여부와 상관없이 블록이
    실제로 가진 이미지를 그대로 보여줘야 하기 때문이다. build_pages()가
    쓰는 것과 동일한 content-addressed image_id 스킴(md5 hexdigest[:16])을
    써서 기존 /documents/{id}/images/{image_id} 서빙 엔드포인트를 그대로
    쓸 수 있게 한다."""
    img_path = block.get("img_path")
    if not img_path:
        return None
    try:
        raw = _decode_data_uri(_lookup_image(images, img_path))
    except (KeyError, ValueError):
        return None
    image_id = hashlib.md5(raw).hexdigest()[:16]
    image = _bgr_array_from_bytes(raw)
    return image_id if _save_image(image, image_dir, image_id) else None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && python -m pytest tests/test_mineru_client.py -k persist_block_image -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/mineru_client.py backend/tests/test_mineru_client.py
git commit -m "feat: add persist_block_image for unconditional block image saving"
```

- [ ] **Step 6: 실패하는 테스트 작성 — `on_parsed` 콜백**

`backend/tests/test_mineru_client.py` 맨 끝에 추가:

```python
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
```

- [ ] **Step 7: 테스트 실패 확인**

Run: `cd backend && python -m pytest tests/test_mineru_client.py -k on_parsed -v`
Expected: FAIL with `TypeError: parse_and_build_pages() got an unexpected keyword argument 'on_parsed'`

- [ ] **Step 8: `on_parsed` 파라미터 구현**

`backend/app/services/mineru_client.py`의 `parse_and_build_pages` 정의(278-294번째 줄)를 다음으로 교체:

```python
def parse_and_build_pages(
    file_path: str,
    ocr: Optional[SupportsImageOcr],
    image_dir: Optional[Path] = None,
    client: Optional[MineruClient] = None,
    on_parsed: Optional[Callable[["MineruResult"], None]] = None,
) -> list:
    """Parse a PDF via MinerU and assemble it into PdfPage objects.

    ocr=None skips OCR augmentation of image blocks entirely (see
    build_pages) - the caller (document_processor.load_pdf) decides this
    based on settings.ocr_enabled. on_parsed, if given, is called with the
    raw MineruResult right after parsing succeeds and before build_pages()
    consumes it - this is the only point callers can access the untouched
    blocks (used by parsed_store.save() for the parsed-result viewer).
    """
    if client is None:
        client = MineruClient()

    result = client.parse_pdf(file_path)
    if on_parsed is not None:
        on_parsed(result)
    return build_pages(result.blocks, result.images, ocr, image_dir)
```

- [ ] **Step 9: 테스트 통과 확인**

Run: `cd backend && python -m pytest tests/test_mineru_client.py -v`
Expected: PASS (all tests in the file, including the two new ones)

- [ ] **Step 10: 커밋**

```bash
git add backend/app/services/mineru_client.py backend/tests/test_mineru_client.py
git commit -m "feat: add on_parsed callback to parse_and_build_pages"
```

---

## Task 2: `parsed_store.py` — 원본 블록 JSON 영속화

**Files:**
- Create: `backend/app/services/parsed_store.py`
- Test: `backend/tests/test_parsed_store.py`
- Modify: `backend/app/config.py`

**Interfaces:**
- Consumes: `persist_block_image(block, images, image_dir) -> Optional[str]`, `MineruResult`(`blocks: list`, `images: dict`) — both from Task 1 (`app.services.mineru_client`).
- Produces: `save(document_id: str, filename: str, result: MineruResult, image_dir: Path) -> None` — never raises. Writes `{settings.parsed_storage_dir}/{document_id}.json` with shape `{document_id, filename, created_at, page_count, pages: [{page_number, blocks: [{type, text?, table_body?, image_id?}]}]}`.

- [ ] **Step 1: 설정 추가**

`backend/app/config.py`의 `mineru_lang_list` 줄(61번째 줄) 다음에 추가:

```python
    mineru_lang_list: list[str] = ["korean"]  # passed verbatim as /file_parse's lang_list form field

    # Parsed-result viewer: raw MinerU content_list blocks, saved alongside images
    parsed_storage_dir: str = "app/storage/parsed"
```

- [ ] **Step 2: 실패하는 테스트 작성**

`backend/tests/test_parsed_store.py` 새로 작성:

```python
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
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd backend && python -m pytest tests/test_parsed_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.parsed_store'`

- [ ] **Step 4: `parsed_store.py` 구현**

`backend/app/services/parsed_store.py` 새로 작성:

```python
"""Persists MinerU's raw content_list blocks per document, for the
parsed-result viewer.

See docs/superpowers/specs/2026-08-31-mineru-result-viewer-design.md. This
is a read-only side channel off document_processor.load_pdf: a failure here
must never break the upload/parsing pipeline, so save() never raises.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.mineru_client import MineruResult, persist_block_image

logger = logging.getLogger(__name__)
settings = get_settings()


def save(document_id: str, filename: str, result: MineruResult, image_dir: Path) -> None:
    """Persist result's raw blocks as {parsed_storage_dir}/{document_id}.json.

    Never raises - any failure (disk full, unwritable path, malformed
    block) is logged and swallowed so it can never break the caller's
    upload/parsing flow.
    """
    try:
        _save(document_id, filename, result, image_dir)
    except Exception as e:
        logger.warning(
            "Failed to persist parsed result for %s (%s): %s", filename, document_id, e
        )


def _save(document_id: str, filename: str, result: MineruResult, image_dir: Path) -> None:
    by_page: dict[int, list] = {}
    for block in result.blocks:
        by_page.setdefault(block.get("page_idx", 0), []).append(block)

    pages = [
        {
            "page_number": page_idx + 1,
            "blocks": [
                _serialize_block(block, result.images, image_dir) for block in by_page[page_idx]
            ],
        }
        for page_idx in sorted(by_page)
    ]

    document = {
        "document_id": document_id,
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages,
    }

    parsed_dir = Path(settings.parsed_storage_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{document_id}.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )


def _serialize_block(block: dict, images: dict, image_dir: Path) -> dict:
    """블록을 원본 그대로 직렬화한다 - build_pages()의 _needs_ocr 같은
    해석 없이, text/table_body/img_path가 있으면 있는 그대로 낸다(design
    doc 3.4절: 표 블록이 table_body와 image_id를 동시에 가질 수 있는
    이유)."""
    out: dict[str, Any] = {"type": block.get("type", "text")}

    text = (block.get("text") or "").strip()
    if text:
        out["text"] = text

    table_body = (block.get("table_body") or "").strip()
    if table_body:
        out["table_body"] = table_body

    image_id = persist_block_image(block, images, image_dir)
    if image_id is not None:
        out["image_id"] = image_id

    return out
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && python -m pytest tests/test_parsed_store.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: 커밋**

```bash
git add backend/app/config.py backend/app/services/parsed_store.py backend/tests/test_parsed_store.py
git commit -m "feat: add parsed_store to persist MinerU's raw blocks per document"
```

---

## Task 3: `document_processor.py` — `load_pdf`에 저장 연결

**Files:**
- Modify: `backend/app/services/document_processor.py`
- Test: `backend/tests/test_document_processor_parsed_store.py`

**Interfaces:**
- Consumes: `parsed_store.save(document_id, filename, result, image_dir)` (Task 2), `parse_and_build_pages(..., on_parsed=...)` (Task 1).
- Produces: `DocumentProcessor.load_pdf(file_path, filename, document_id=None)`이 `document_id`가 주어졌을 때 `app/storage/parsed/{document_id}.json`을 부수효과로 남긴다는 계약(하위 태스크 없음 - 이 문서 시스템의 최종 소비자는 API 라우트).

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_document_processor_parsed_store.py` 새로 작성:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && python -m pytest tests/test_document_processor_parsed_store.py -v`
Expected: `test_parsed_json_is_written_when_document_id_given`, `test_mineru_disabled_skips_parsed_json_too`(방식은 아래 참고)는 통과하지 않음 — 구체적으로 첫 번째 테스트가 `AssertionError` (parsed_path.exists()가 False)로 FAIL. 나머지는 아직 아무것도 안 하므로 우연히 통과할 수 있다.

- [ ] **Step 3: `load_pdf` 연결 구현**

`backend/app/services/document_processor.py`의 import 블록(11-13번째 줄)을 다음으로 교체:

```python
from app.config import get_settings
from app.services.chunking import build_splitter
from app.services.mineru_client import PdfPage, parse_and_build_pages
from app.services import parsed_store
```

`load_pdf` 메서드(28-96번째 줄) 안의 다음 블록(64-73번째 줄):

```python
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
```

를 다음으로 교체(`try:` 앞에 `on_parsed` 결정 로직 두 줄을 추가하고, 호출에 `on_parsed=on_parsed` 인자만 더한 것 — `except` 블록은 완전히 동일):

```python
        on_parsed = None
        if document_id:
            on_parsed = lambda result: parsed_store.save(document_id, filename, result, image_dir)

        try:
            pages: list[PdfPage] = parse_and_build_pages(
                file_path, ocr=ocr, image_dir=image_dir, client=self.mineru_client, on_parsed=on_parsed
            )
        except Exception as e:
            logger.warning(
                f"MinerU extraction failed for {filename} ({e}); "
                "falling back to text-only extraction"
            )
            return PyPDFLoader(file_path).load()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && python -m pytest tests/test_document_processor_parsed_store.py tests/test_document_processor_ocr.py tests/test_mineru_client.py -v`
Expected: PASS (전부 — 기존 `test_document_processor_ocr.py`가 회귀되지 않았음을 함께 확인)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/document_processor.py backend/tests/test_document_processor_parsed_store.py
git commit -m "feat: wire parsed_store into DocumentProcessor.load_pdf"
```

---

## Task 4: 백엔드 조회 API

**Files:**
- Modify: `backend/app/api/models/responses.py`
- Modify: `backend/app/api/routes/documents.py`
- Test: `backend/tests/test_documents_parsed_api.py`

**Interfaces:**
- Consumes: `settings.parsed_storage_dir`(Task 2)에 Task 3이 써 놓는 JSON 파일 형식(`{document_id, filename, created_at, page_count, pages: [{page_number, blocks: [{type, text?, table_body?, image_id?}]}]}`).
- Produces: `GET /api/v1/documents/parsed` → `List[ParsedDocumentSummary]`, `GET /api/v1/documents/{document_id}/parsed` → `ParsedDocumentDetail`(404 시 `HTTPException`). 프론트엔드(Task 5)가 그대로 소비할 JSON 응답 shape.

- [ ] **Step 1: 응답 모델 추가**

`backend/app/api/models/responses.py` 맨 끝(66번째 줄)에 추가:

```python


class ParsedBlock(BaseModel):
    """MinerU content_list 블록을 원본 그대로 옮긴 것."""
    type: str
    text: Optional[str] = None
    table_body: Optional[str] = None
    image_id: Optional[str] = None


class ParsedPage(BaseModel):
    page_number: int
    blocks: list[ParsedBlock]


class ParsedDocumentSummary(BaseModel):
    """파싱 결과 목록 화면에 쓰이는 요약."""
    document_id: str
    filename: str
    created_at: str
    page_count: int


class ParsedDocumentDetail(ParsedDocumentSummary):
    """파싱 결과 상세 - 페이지별 원본 블록 전체."""
    pages: list[ParsedPage]
```

- [ ] **Step 2: 실패하는 테스트 작성**

`backend/tests/test_documents_parsed_api.py` 새로 작성 (기존 `test_document_images_api.py`처럼 TestClient 없이 라우트 함수를 직접 호출하는 패턴을 따른다):

```python
"""파싱 결과 조회 라우트(목록/상세) 테스트.

test_document_images_api.py와 같은 패턴 - route 핸들러가 위임하는 순수
함수를 직접 호출한다 (TestClient 불필요)."""
import json

import pytest
from fastapi import HTTPException

from app.api.routes.documents import list_parsed_documents, load_parsed_document


def _write_parsed(dir_, document_id, filename="a.pdf", page_count=1, pages=None):
    dir_.mkdir(parents=True, exist_ok=True)
    data = {
        "document_id": document_id,
        "filename": filename,
        "created_at": "2026-08-31T00:00:00+00:00",
        "page_count": page_count,
        "pages": pages or [{"page_number": 1, "blocks": [{"type": "text", "text": "hi"}]}],
    }
    (dir_ / f"{document_id}.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def test_list_parsed_documents_returns_summaries_for_each_file(tmp_path):
    _write_parsed(tmp_path, "doc_a", filename="a.pdf", page_count=3)
    _write_parsed(tmp_path, "doc_b", filename="b.pdf", page_count=1)

    result = list_parsed_documents(tmp_path)

    ids = {s.document_id for s in result}
    assert ids == {"doc_a", "doc_b"}


def test_list_parsed_documents_returns_empty_list_when_dir_missing(tmp_path):
    result = list_parsed_documents(tmp_path / "missing")

    assert result == []


def test_list_parsed_documents_skips_unreadable_files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")
    _write_parsed(tmp_path, "doc_ok")

    result = list_parsed_documents(tmp_path)

    assert [s.document_id for s in result] == ["doc_ok"]


def test_load_parsed_document_returns_full_detail(tmp_path):
    _write_parsed(tmp_path, "doc_a")

    result = load_parsed_document("doc_a", tmp_path)

    assert result.document_id == "doc_a"
    assert result.pages[0].blocks[0].text == "hi"


def test_load_parsed_document_raises_404_when_missing(tmp_path):
    with pytest.raises(HTTPException) as exc:
        load_parsed_document("doc_missing", tmp_path)

    assert exc.value.status_code == 404


@pytest.mark.parametrize("document_id", ["..", "../etc", "doc/../..", "doc abc"])
def test_load_parsed_document_rejects_unsafe_ids(document_id, tmp_path):
    with pytest.raises(HTTPException) as exc:
        load_parsed_document(document_id, tmp_path)

    assert exc.value.status_code == 404
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd backend && python -m pytest tests/test_documents_parsed_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'list_parsed_documents'`

- [ ] **Step 4: 라우트 구현**

`backend/app/api/routes/documents.py`의 import 블록(1-13번째 줄)을 다음으로 교체:

```python
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import json
import logging
import re
import shutil
from typing import List

from app.api.models.responses import (
    DocumentInfo,
    ParsedDocumentDetail,
    ParsedDocumentSummary,
)
from app.config import get_settings
```

`documents.py`의 다음 블록(58-66번째 줄):

```python
@router.get("/{document_id}/images/{image_id}")
async def get_document_image(document_id: str, image_id: str) -> FileResponse:
    """문서에서 추출된 도표 이미지를 서빙한다."""
    path = resolve_image_path(document_id, image_id, Path(settings.image_storage_dir))
    return FileResponse(path, media_type="image/png")


@router.get("/", response_model=List[DocumentInfo])
```

를 다음으로 교체(새 함수/라우트 2개를 그 사이에 삽입하고, 마지막 데코레이터 줄은 그대로 유지):

```python
@router.get("/{document_id}/images/{image_id}")
async def get_document_image(document_id: str, image_id: str) -> FileResponse:
    """문서에서 추출된 도표 이미지를 서빙한다."""
    path = resolve_image_path(document_id, image_id, Path(settings.image_storage_dir))
    return FileResponse(path, media_type="image/png")


def list_parsed_documents(parsed_dir: Path) -> List[ParsedDocumentSummary]:
    """parsed_dir의 모든 파싱 결과 파일에서 요약 정보만 읽어 목록으로
    반환한다. 문서 삭제와 파싱 JSON 삭제는 별개이므로(design doc 4절),
    parsed_dir에 실제로 존재하는 파일이 곧 목록이다."""
    if not parsed_dir.is_dir():
        return []

    summaries = []
    for path in sorted(parsed_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                ParsedDocumentSummary(
                    document_id=data["document_id"],
                    filename=data["filename"],
                    created_at=data["created_at"],
                    page_count=data["page_count"],
                )
            )
        except Exception:
            logger.warning("Skipping unreadable parsed result file: %s", path)
            continue

    return summaries


def load_parsed_document(document_id: str, parsed_dir: Path) -> ParsedDocumentDetail:
    """document_id의 파싱 결과 상세를 읽는다. document_id는 사용자 입력이
    파일 경로가 되므로 resolve_image_path와 동일하게 화이트리스트로
    검증한다."""
    if not _SAFE_ID.match(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parsed result not found")

    path = parsed_dir / f"{document_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parsed result not found")

    data = json.loads(path.read_text(encoding="utf-8"))
    return ParsedDocumentDetail(**data)


@router.get("/parsed", response_model=List[ParsedDocumentSummary])
async def get_parsed_documents() -> List[ParsedDocumentSummary]:
    """MinerU로 파싱된 문서 목록을 반환한다."""
    return list_parsed_documents(Path(settings.parsed_storage_dir))


@router.get("/{document_id}/parsed", response_model=ParsedDocumentDetail)
async def get_parsed_document(document_id: str) -> ParsedDocumentDetail:
    """한 문서의 MinerU 원본 파싱 결과(페이지별 블록)를 반환한다."""
    return load_parsed_document(document_id, Path(settings.parsed_storage_dir))


@router.get("/", response_model=List[DocumentInfo])
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && python -m pytest tests/test_documents_parsed_api.py tests/test_document_images_api.py -v`
Expected: PASS (전부 — 기존 이미지 API 테스트가 회귀되지 않았음을 함께 확인)

- [ ] **Step 6: 전체 백엔드 테스트 스위트 실행**

Run: `cd backend && python -m pytest -v`
Expected: PASS (모든 기존 테스트 + 이번 태스크에서 추가한 테스트)

- [ ] **Step 7: 커밋**

```bash
git add backend/app/api/models/responses.py backend/app/api/routes/documents.py backend/tests/test_documents_parsed_api.py
git commit -m "feat: add parsed-result list/detail API endpoints"
```

---

## Task 5: 프론트엔드 — 타입 + API 클라이언트

**Files:**
- Modify: `frontend/src/types/api.types.ts`
- Modify: `frontend/src/services/api.ts`

**Interfaces:**
- Consumes: Task 4의 응답 shape (`ParsedDocumentSummary`, `ParsedDocumentDetail` JSON).
- Produces: `apiService.getParsedDocuments(): Promise<ParsedDocumentSummary[]>`, `apiService.getParsedDocument(documentId: string): Promise<ParsedDocumentDetail>` — Task 6이 그대로 호출.

이 태스크는 순수 타입/네트워크 계층 추가라 기존 프론트엔드에 테스트 프레임워크가 없으므로(저장소에 `*.test.*`/vitest/jest 없음 확인됨), TDD 대신 타입체크로 검증한다.

- [ ] **Step 1: 타입 추가**

`frontend/src/types/api.types.ts` 맨 끝(51번째 줄)에 추가:

```typescript

export interface ParsedBlock {
  type: string;
  text?: string;
  table_body?: string;
  image_id?: string;
}

export interface ParsedPage {
  page_number: number;
  blocks: ParsedBlock[];
}

export interface ParsedDocumentSummary {
  document_id: string;
  filename: string;
  created_at: string;
  page_count: number;
}

export interface ParsedDocumentDetail extends ParsedDocumentSummary {
  pages: ParsedPage[];
}
```

- [ ] **Step 2: API 클라이언트 메서드 추가**

`frontend/src/services/api.ts`의 import 줄(2번째 줄)을 다음으로 교체:

```typescript
import { ChatRequest, ChatResponse, UploadResponse, UploadOptions, DocumentInfo, ParsedDocumentSummary, ParsedDocumentDetail } from '../types/api.types';
```

`deleteDocument` 메서드(135-137번째 줄) 다음, 클래스 닫는 중괄호 앞에 추가:

```typescript

  /**
   * Get list of MinerU-parsed documents (raw block results)
   */
  async getParsedDocuments(): Promise<ParsedDocumentSummary[]> {
    const response = await this.client.get<ParsedDocumentSummary[]>('/documents/parsed');
    return response.data;
  }

  /**
   * Get one document's raw parsed blocks (page by page)
   */
  async getParsedDocument(documentId: string): Promise<ParsedDocumentDetail> {
    const response = await this.client.get<ParsedDocumentDetail>(`/documents/${documentId}/parsed`);
    return response.data;
  }
```

- [ ] **Step 3: 타입체크로 검증**

Run: `cd frontend && npx tsc --noEmit`
Expected: 에러 없음 (기존 에러가 있었다면 그대로 — 이번 변경으로 새 에러가 추가되지 않았는지만 확인)

- [ ] **Step 4: 커밋**

```bash
git add frontend/src/types/api.types.ts frontend/src/services/api.ts
git commit -m "feat: add frontend types and API client methods for parsed results"
```

---

## Task 6: 프론트엔드 — `ParsedDocumentViewer` 컴포넌트

**Files:**
- Create: `frontend/src/components/ParsedDocumentViewer.tsx`
- Create: `frontend/src/styles/ParsedDocumentViewer.css`
- Modify: `frontend/package.json`

**Interfaces:**
- Consumes: `apiService.getParsedDocuments()`, `apiService.getParsedDocument(documentId)`(Task 5), `ParsedDocumentSummary`, `ParsedDocumentDetail`, `ParsedBlock`, `ParsedPage`(Task 5 타입).
- Produces: `export const ParsedDocumentViewer: React.FC` — Task 7(`App.tsx`)이 그대로 렌더링.

- [ ] **Step 1: `dompurify` 의존성 추가**

Run: `cd frontend && npm install dompurify && npm install --save-dev @types/dompurify`

Expected: `frontend/package.json`의 `dependencies`에 `dompurify`, `devDependencies`에 `@types/dompurify` 추가됨.

- [ ] **Step 2: 컴포넌트 작성**

`frontend/src/components/ParsedDocumentViewer.tsx` 새로 작성:

```tsx
import React, { useEffect, useState } from 'react';
import DOMPurify from 'dompurify';
import { FileText } from 'lucide-react';
import { apiService } from '../services/api';
import { ParsedBlock, ParsedDocumentDetail, ParsedDocumentSummary } from '../types/api.types';
import '../styles/ParsedDocumentViewer.css';

function ParsedBlockView({ documentId, block }: { documentId: string; block: ParsedBlock }) {
  if (block.table_body) {
    return (
      <div
        className="parsed-block parsed-block-table"
        dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(block.table_body) }}
      />
    );
  }

  if (block.type === 'equation' && block.text) {
    return (
      <pre className="parsed-block parsed-block-equation">
        <code>{block.text}</code>
      </pre>
    );
  }

  if (block.image_id) {
    return (
      <img
        className="parsed-block parsed-block-image"
        src={`/api/v1/documents/${documentId}/images/${block.image_id}`}
        alt={`${block.type} 블록`}
        loading="lazy"
      />
    );
  }

  if (block.text) {
    const isTitle = block.type === 'title';
    return (
      <p className={isTitle ? 'parsed-block parsed-block-title' : 'parsed-block parsed-block-text'}>
        {block.text}
      </p>
    );
  }

  return null;
}

export const ParsedDocumentViewer: React.FC = () => {
  const [documents, setDocuments] = useState<ParsedDocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ParsedDocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiService
      .getParsedDocuments()
      .then(setDocuments)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    setError(null);
    apiService
      .getParsedDocument(selectedId)
      .then(setDetail)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedId]);

  return (
    <div className="parsed-viewer">
      <div className="parsed-viewer-sidebar">
        <h2>파싱된 문서</h2>
        {documents.length === 0 && <p className="parsed-viewer-empty">아직 파싱된 문서가 없습니다.</p>}
        <ul>
          {documents.map((doc) => (
            <li key={doc.document_id}>
              <button
                className={doc.document_id === selectedId ? 'parsed-doc-item active' : 'parsed-doc-item'}
                onClick={() => setSelectedId(doc.document_id)}
              >
                <FileText size={14} />
                <span className="parsed-doc-filename">{doc.filename}</span>
                <span className="parsed-doc-pages">{doc.page_count}p</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="parsed-viewer-content">
        {error && <p className="parsed-viewer-error">{error}</p>}
        {loading && <p>불러오는 중...</p>}
        {!loading && !error && !detail && <p className="parsed-viewer-empty">왼쪽에서 문서를 선택하세요.</p>}
        {detail &&
          detail.pages.map((page) => (
            <section key={page.page_number} className="parsed-page">
              <h3>Page {page.page_number}</h3>
              {page.blocks.map((block, idx) => (
                <ParsedBlockView key={idx} documentId={detail.document_id} block={block} />
              ))}
            </section>
          ))}
      </div>
    </div>
  );
};
```

- [ ] **Step 3: 스타일 작성**

`frontend/src/styles/ParsedDocumentViewer.css` 새로 작성:

```css
.parsed-viewer {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 1.5rem;
  height: 100%;
  min-height: 0;
}

.parsed-viewer-sidebar {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
}

.parsed-viewer-sidebar h2 {
  margin: 0 0 1rem;
  font-size: 1.1rem;
}

.parsed-viewer-sidebar ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.parsed-doc-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  text-align: left;
  font-size: 0.9rem;
}

.parsed-doc-item.active {
  border-color: #667eea;
  background: #f0f1fd;
}

.parsed-doc-filename {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.parsed-doc-pages {
  color: #888;
  font-size: 0.8rem;
}

.parsed-viewer-content {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
}

.parsed-viewer-empty {
  color: #888;
}

.parsed-viewer-error {
  color: #c0392b;
}

.parsed-page {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid #eee;
}

.parsed-page h3 {
  margin: 0 0 1rem;
  color: #667eea;
}

.parsed-block {
  margin: 0 0 1rem;
}

.parsed-block-title {
  font-weight: 700;
}

.parsed-block-table table {
  border-collapse: collapse;
  width: 100%;
}

.parsed-block-table td,
.parsed-block-table th {
  border: 1px solid #ddd;
  padding: 0.4rem 0.6rem;
}

.parsed-block-equation {
  background: #f5f5f5;
  padding: 0.75rem;
  border-radius: 6px;
  overflow-x: auto;
}

.parsed-block-image {
  max-width: 100%;
  border-radius: 6px;
}
```

- [ ] **Step 4: 타입체크로 검증**

Run: `cd frontend && npx tsc --noEmit`
Expected: 에러 없음

- [ ] **Step 5: 커밋**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/components/ParsedDocumentViewer.tsx frontend/src/styles/ParsedDocumentViewer.css
git commit -m "feat: add ParsedDocumentViewer component"
```

---

## Task 7: 프론트엔드 — `App.tsx` 탭 연결 + 수동 검증

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles/App.css`

**Interfaces:**
- Consumes: `ParsedDocumentViewer`(Task 6).
- Produces: 없음(최종 사용자 화면).

- [ ] **Step 1: 탭 토글 구현**

`frontend/src/App.tsx` 전체를 다음으로 교체:

```tsx
import { useState } from 'react';
import { FileUploader } from './components/FileUploader';
import { ChatWindow } from './components/ChatWindow';
import { ParsedDocumentViewer } from './components/ParsedDocumentViewer';
import './styles/App.css';

type ActiveTab = 'chat' | 'viewer';

function App() {
  const [uploadedDocumentIds, setUploadedDocumentIds] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<ActiveTab>('chat');

  const handleUploadComplete = (documentIds: string[]) => {
    setUploadedDocumentIds((prev) => [...prev, ...documentIds]);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Knowledge Assist RAG</h1>
        <p>Upload documents and chat with them using AI</p>
        <nav className="app-tabs">
          <button
            className={activeTab === 'chat' ? 'app-tab active' : 'app-tab'}
            onClick={() => setActiveTab('chat')}
          >
            채팅
          </button>
          <button
            className={activeTab === 'viewer' ? 'app-tab active' : 'app-tab'}
            onClick={() => setActiveTab('viewer')}
          >
            문서 뷰어
          </button>
        </nav>
      </header>

      <main className="app-main">
        {activeTab === 'chat' ? (
          <>
            <div className="sidebar">
              <FileUploader onUploadComplete={handleUploadComplete} />
            </div>
            <div className="chat-section">
              <ChatWindow documentIds={uploadedDocumentIds} />
            </div>
          </>
        ) : (
          <div className="app-main-full">
            <ParsedDocumentViewer />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
```

- [ ] **Step 2: 탭 + 전체폭 레이아웃 스타일 추가**

`frontend/src/styles/App.css` 맨 끝(59번째 줄)에 추가:

```css

.app-tabs {
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  margin-top: 1rem;
}

.app-tab {
  padding: 0.5rem 1.25rem;
  border: 1px solid rgba(255, 255, 255, 0.6);
  border-radius: 999px;
  background: transparent;
  color: white;
  cursor: pointer;
  font-size: 0.95rem;
}

.app-tab.active {
  background: white;
  color: #667eea;
  font-weight: 600;
}

.app-main-full {
  grid-column: 1 / -1;
  min-height: 0;
}
```

- [ ] **Step 3: 타입체크로 검증**

Run: `cd frontend && npx tsc --noEmit`
Expected: 에러 없음

- [ ] **Step 4: dev 서버로 수동 검증**

Run: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (백그라운드)
Run: `cd frontend && npm run dev` (백그라운드)

브라우저에서 확인:
1. PDF 한 건 업로드 → "채팅" 탭에서 정상 동작 확인(회귀 없음).
2. "문서 뷰어" 탭 클릭 → 방금 업로드한 문서가 좌측 목록에 나타나는지 확인.
3. 문서 클릭 → 우측에 페이지별 블록(텍스트/표/이미지)이 렌더링되는지 확인. 표가 있는 PDF라면 `<table>`이 브라우저 기본 스타일로라도 보이는지, 이미지가 있다면 `<img>`가 실제로 로드되는지 확인.
4. `mineru_enabled=False`인 문서(또는 업로드 실패로 폴백된 문서)는 파싱 결과가 없으므로 목록에 나타나지 않아야 함 — 필요시 `.env`에서 `MINERU_ENABLED=false`로 잠깐 바꿔 업로드해보고 목록에 안 나타나는지 확인.

Expected: 위 4가지 시나리오 모두 의도대로 동작.

- [ ] **Step 5: 커밋**

```bash
git add frontend/src/App.tsx frontend/src/styles/App.css
git commit -m "feat: add document viewer tab to App"
```

---

## Self-Review Notes

- **스펙 커버리지**: 3.2(저장) → Task 1-3, 3.3(API) → Task 4, 3.4(프론트엔드) → Task 5-7, 3.5(설정) → Task 2 Step 1. 4절 비대상(LaTeX 렌더링 안 함, 문서 삭제 시 파싱 JSON 정리 안 함, 페이지네이션 없음)은 어느 태스크에도 포함하지 않음 — 의도된 누락.
- **타입 일관성**: `ParsedBlock{type, text?, table_body?, image_id?}`가 Task 4(Pydantic)와 Task 5(TypeScript)에서 필드명이 정확히 일치함을 확인함. `parsed_store._serialize_block`이 만드는 dict 키(`type`/`text`/`table_body`/`image_id`)도 동일.
- **플레이스홀더 스캔**: "TODO"/"나중에" 등 없음. 모든 스텝에 실행 가능한 코드 포함.
