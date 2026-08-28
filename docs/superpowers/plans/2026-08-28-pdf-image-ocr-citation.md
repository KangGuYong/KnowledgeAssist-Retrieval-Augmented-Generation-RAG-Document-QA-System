# PDF 도표 출처 인라인 이미지 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 검색된 청크가 PDF 도표 이미지에서 OCR로 추출된 것일 때, 그 페이지에서 인식된
원본 이미지를 채팅 출처 카드 안에 인라인으로 표시한다.

**Architecture:** OCR 텍스트 인덱싱 자체는 이미 `pdf_ocr.py`(PaddleOCR, 서브프로세스
격리)로 구현되어 있다. 이 계획은 그 파이프라인이 이미지를 렌더링해 OCR에 넘기는
지점에서 PNG로도 저장하고, 페이지 단위로 이미지 ID를 추적해 청크 메타데이터에
실어 나르고, 새 엔드포인트로 서빙하고, `SourceCitation`이 인라인으로 그린다.
청크-이미지 연결은 페이지 단위 근사다(설계 문서 3.1절).

**Tech Stack:** FastAPI, LangChain 0.1.0, ChromaDB, PyMuPDF(fitz), PaddleOCR
(서브프로세스), React 18 + TypeScript, Vite

**설계 문서:** `docs/superpowers/specs/2026-08-28-pdf-image-ocr-citation-design.md`

**이 계획이 대체하는 것:** 동일 파일명의 이전 버전은 EasyOCR·별도 이미지 청크를
전제로 `image_text_extractor.py`라는 신규 모듈을 만드는 계획이었다. 그 사이
`pdf_ocr.py`/`ocr_service.py`/`ocr_worker.py`가 다른 아키텍처로 이미 구현·테스트되어
있어 그 계획은 그대로 실행하면 기존 코드와 충돌한다(`requirements.txt`에 EasyOCR과
PaddleOCR을 동시에 넣게 되고, 이미 통과 중인 `test_document_processor_ocr.py`를
깨뜨린다). 이 버전은 실제 코드를 확장하는 쪽으로 전면 재작성했다.

**테스트 관례:** 이 저장소의 기존 OCR 테스트(`test_pdf_ocr.py`,
`test_document_processor_ocr.py`)는 `conftest.py` 픽스처를 쓰지 않고, 파일마다
`fitz`로 합성 PDF를 만드는 헬퍼 함수와 `StubOCR` 클래스를 직접 정의한다. 아래 태스크도
그 관례를 따른다 — 새 `conftest.py`를 만들지 않는다.

---

## 파일 구조

**신규**

| 파일 | 책임 |
|---|---|
| `backend/tests/test_document_images_api.py` | 이미지 서빙 엔드포인트 및 경로 검증 테스트 |

**수정**

| 파일 | 변경 |
|---|---|
| `backend/app/config.py` | `image_storage_dir` 설정 1개 추가 |
| `backend/app/services/pdf_ocr.py` | `PdfPage.image_ids` 추가, 이미지 저장 로직, `extract_pages`/`_extract_page`에 `image_dir` 매개변수 |
| `backend/app/services/document_processor.py` | `load_pdf`/`load_document`/`process_file`에 `document_id` 추가, `image_ids` 메타데이터 승계 |
| `backend/app/api/routes/upload.py` | `process_file` 호출에 `document_id` 전달 |
| `backend/app/api/routes/documents.py` | 이미지 서빙 엔드포인트 추가, 삭제 시 이미지 정리 |
| `backend/app/api/models/responses.py` | `SourceDocument`에 `image_urls` 추가 |
| `backend/app/services/rag_service.py` | `_format_sources`에 `image_urls`, OCR 주의 프롬프트 |
| `backend/tests/test_pdf_ocr.py` | 이미지 저장 테스트 추가 |
| `backend/tests/test_document_processor_ocr.py` | `image_ids` 메타데이터 승계 테스트 추가 |
| `frontend/src/types/api.types.ts` | `SourceDocument`에 `image_urls: string[]` |
| `frontend/src/components/SourceCitation.tsx` | 인라인 썸네일 + 라이트박스 |
| `frontend/src/styles/SourceCitation.css` | 썸네일/라이트박스 스타일 |

---

## Task 1: 설정 추가

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/tests/test_ocr_service.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`backend/tests/test_ocr_service.py`의 `test_defaults_use_the_korean_ppocrv5_recogniser` 근처에 추가한다.

```python
def test_image_storage_dir_defaults_alongside_other_storage_paths():
    settings = Settings()

    assert settings.image_storage_dir == "app/storage/images"
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && python -m pytest tests/test_ocr_service.py -v -k image_storage_dir`
Expected: `AttributeError: 'Settings' object has no attribute 'image_storage_dir'`

- [ ] **Step 3: `Settings`에 필드 추가**

`backend/app/config.py`의 `ocr_empty_placeholder` 줄 바로 아래, OCR 섹션 안에 추가한다.

```python
    image_storage_dir: str = "app/storage/images"  # 도표 이미지 저장 경로
```

- [ ] **Step 4: 통과 확인 및 커밋**

Run: `cd backend && python -m pytest tests/test_ocr_service.py -v`
Expected: PASS

```bash
git add backend/app/config.py backend/tests/test_ocr_service.py
git commit -m "feat: add image storage directory setting"
```

---

## Task 2: `pdf_ocr.py` — 이미지 저장

이미지를 렌더링해 OCR에 넘기는 지점에서 같은 배열을 PNG로도 저장한다. 새 모듈을
만들지 않고 기존 파일을 확장한다.

**Files:**
- Modify: `backend/app/services/pdf_ocr.py`
- Modify: `backend/tests/test_pdf_ocr.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`backend/tests/test_pdf_ocr.py`에 추가한다(기존 `StubOCR`, `_build_pdf`, `_image_bytes` 헬퍼를 재사용한다).

```python
def test_image_block_is_saved_as_png(tmp_path):
    """블록 단위 이미지는 image_dir에 PNG로 저장되고 page.image_ids에 실린다."""
    image_dir = tmp_path / "images"
    pdf_path = _build_pdf(tmp_path)

    pages = extract_pages(pdf_path, ocr=StubOCR(), image_dir=image_dir)

    assert len(pages[0].image_ids) == 1
    image_id = pages[0].image_ids[0]
    saved = image_dir / f"{image_id}.png"
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_no_image_dir_skips_saving_without_failing(tmp_path):
    """image_dir=None이면(기존 호출부 호환) 저장을 건너뛰고 텍스트 추출은 그대로 동작한다."""
    pdf_path = _build_pdf(tmp_path)

    pages = extract_pages(pdf_path, ocr=StubOCR())

    assert "스캔된 표 내용" in pages[0].text
    assert pages[0].image_ids == []


def test_repeated_image_across_pages_is_saved_once(tmp_path):
    """같은 이미지 바이트가 여러 페이지에 반복돼도 같은 image_id, 같은 파일로 수렴한다."""
    doc = fitz.open()
    image_bytes = _image_bytes()
    for _ in range(2):
        page = doc.new_page()
        page.insert_image(fitz.Rect(72, 200, 300, 380), stream=image_bytes)
    pdf_path = tmp_path / "repeated.pdf"
    doc.save(pdf_path)
    doc.close()
    image_dir = tmp_path / "images"

    pages = extract_pages(str(pdf_path), ocr=StubOCR(), image_dir=image_dir)

    assert pages[0].image_ids == pages[1].image_ids
    assert len(list(image_dir.glob("*.png"))) == 1


def test_image_save_failure_does_not_break_text_extraction(tmp_path):
    """저장 디렉터리를 쓸 수 없어도 OCR 텍스트 추출은 실패하지 않는다."""
    pdf_path = _build_pdf(tmp_path)
    unwritable = tmp_path / "not_a_directory"
    unwritable.write_text("occupied")  # 디렉터리로 만들 수 없는 경로

    pages = extract_pages(pdf_path, ocr=StubOCR(), image_dir=unwritable)

    assert "스캔된 표 내용" in pages[0].text
    assert pages[0].image_ids == []


def test_full_page_scan_is_saved_with_a_page_scoped_id(tmp_path, monkeypatch):
    """전체 페이지 OCR(스캔본/이미지 과다)도 하나의 이미지로 저장된다."""
    monkeypatch.setattr(pdf_ocr.settings, "ocr_page_text_threshold", 10_000)  # 항상 스캔으로 판정
    pdf_path = _build_pdf(tmp_path)
    image_dir = tmp_path / "images"

    pages = extract_pages(pdf_path, ocr=StubOCR(), image_dir=image_dir)

    assert pages[0].full_page_ocr is True
    assert pages[0].image_ids == ["p1_full"]
    assert (image_dir / "p1_full.png").exists()
```

상단 import에 `from app.services import pdf_ocr`가 없다면 추가한다(2번째 테스트가 `pdf_ocr.settings`를 통해 monkeypatch한다).

- [ ] **Step 2: 실패 확인**

Run: `cd backend && python -m pytest tests/test_pdf_ocr.py -v -k "image_dir or image_ids or saved or scoped"`
Expected: `TypeError: extract_pages() got an unexpected keyword argument 'image_dir'`

- [ ] **Step 3: `pdf_ocr.py` 수정**

`import` 블록에 `field`를 추가한다.

```python
from dataclasses import dataclass, field
```

`PdfPage`에 필드를 추가한다.

```python
@dataclass
class PdfPage:
    """One extracted page, with image regions already replaced by text."""

    page_number: int  # 1-based
    text: str
    image_count: int = 0
    ocr_image_count: int = 0
    full_page_ocr: bool = False
    image_ids: list[str] = field(default_factory=list)
```

`_OcrStats`에 필드를 추가한다.

```python
@dataclass
class _OcrStats:
    images: int = 0
    ocr_images: int = 0
    image_ids: list[str] = field(default_factory=list)
```

이미지 저장 헬퍼를 추가한다. 실패해도 절대 예외를 올리지 않는다 — 호출부에서
try/except를 또 두지 않아도 되게 한다(설계 문서 3.6절).

```python
def _save_image(image: Any, image_dir: Path, image_id: str) -> None:
    """Persist a rendered BGR array as PNG. Never raises; logs and returns on failure."""
    try:
        from PIL import Image as PILImage

        image_dir.mkdir(parents=True, exist_ok=True)
        rgb = image[:, :, ::-1]  # BGR -> RGB, matches _pixmap_to_array's channel order
        PILImage.fromarray(rgb).save(image_dir / f"{image_id}.png", format="PNG")
    except Exception as e:
        logger.warning("Failed to save extracted image %s: %s", image_id, e)
```

`_ocr_image_block`을 교체한다. 캐시가 이제 `(text, image_id)` 쌍을 저장하므로,
캐시 적중 시에는 이미지를 다시 렌더링·저장하지 않고 이미 저장된 파일의 id를 그대로
재사용한다.

```python
def _ocr_image_block(
    page, block: dict, ocr: SupportsImageOcr, cache: dict[str, tuple[str, str]],
    image_dir: Optional[Path],
) -> tuple[str, Optional[str]]:
    """Run OCR on one image block, reusing results and saved PNGs for repeated images.

    Returns (text, image_id). image_id is None only if the block carries no
    raw image bytes to derive a stable id from (should not happen in practice).
    """
    import fitz

    raw = block.get("image")
    key = hashlib.md5(raw).hexdigest()[:16] if raw else None

    if key is not None and key in cache:
        return cache[key]

    image = _render(page, clip=fitz.Rect(block["bbox"]))
    text = (ocr.image_to_text(image) or "").strip()
    image_id = key

    if image_dir is not None and image_id is not None:
        _save_image(image, image_dir, image_id)

    result = (text, image_id)
    if key is not None:
        cache[key] = result
    return result
```

`_extract_page`의 시그니처를 교체한다.

```python
def _extract_page(
    page,
    page_number: int,
    ocr: SupportsImageOcr,
    cache: Optional[dict[str, tuple[str, str]]] = None,
    image_dir: Optional[Path] = None,
) -> PdfPage:
```

전체 페이지 스캔 분기(`looks_scanned or too_many_images`)를 교체한다 — 렌더링된
배열을 변수로 잡아 OCR과 저장에 모두 쓴다.

```python
    if looks_scanned or too_many_images:
        image_id = f"p{page_number}_full"
        try:
            rendered = _render(page)
            text = (ocr.image_to_text(rendered) or "").strip()
        except Exception as e:
            logger.warning("OCR failed on page %s: %s", page_number, e)
            text = ""
            rendered = None
        if not text:
            return PdfPage(
                page_number=page_number,
                text=native_text.strip(),
                image_count=len(image_blocks),
            )
        if image_dir is not None and rendered is not None:
            _save_image(rendered, image_dir, image_id)
        return PdfPage(
            page_number=page_number,
            text=_format_ocr_block(text),
            image_count=len(image_blocks),
            ocr_image_count=len(image_blocks),
            full_page_ocr=True,
            image_ids=[image_id],
        )
```

블록 루프를 교체한다.

```python
    blocks = list(text_blocks)
    for block in image_blocks:
        stats.images += 1
        try:
            text, image_id = _ocr_image_block(page, block, ocr, cache, image_dir)
        except Exception as e:
            logger.warning("OCR failed on page %s: %s", page_number, e)
            text, image_id = "", None

        if not text:
            if settings.ocr_keep_empty_placeholder:
                bbox = block["bbox"]
                blocks.append(
                    _Block(
                        order=(float(bbox[0]), float(bbox[1])),
                        text=settings.ocr_empty_placeholder,
                    )
                )
            continue

        stats.ocr_images += 1
        if image_id is not None:
            stats.image_ids.append(image_id)
        bbox = block["bbox"]
        blocks.append(
            _Block(
                order=(float(bbox[0]), float(bbox[1])),
                text=_format_ocr_block(text),
            )
        )

    ordered = _blocks_in_reading_order(blocks)
    page_text = "\n\n".join(b.text for b in ordered).strip()

    return PdfPage(
        page_number=page_number,
        text=page_text,
        image_count=stats.images,
        ocr_image_count=stats.ocr_images,
        image_ids=stats.image_ids,
    )
```

`extract_pages`의 시그니처와 호출부를 교체한다.

```python
def extract_pages(
    file_path: str,
    ocr: Optional[SupportsImageOcr] = None,
    image_dir: Optional[Path] = None,
) -> list[PdfPage]:
    """Extract every page of a PDF with image regions replaced by OCR text.

    Args:
        file_path: Path to the PDF file.
        ocr: OCR backend to use; defaults to the shared PaddleOCR service.
        image_dir: Where to save recognised images as PNG. None skips saving
            (text extraction is unaffected either way).

    Returns:
        One :class:`PdfPage` per page, in document order.
    """
    import fitz

    if ocr is None:
        from app.services.ocr_service import get_ocr_service

        ocr = get_ocr_service()

    pages: list[PdfPage] = []
    cache: dict[str, tuple[str, str]] = {}
    with fitz.open(file_path) as document:
        for index, page in enumerate(document):
            pages.append(_extract_page(page, index + 1, ocr, cache, image_dir))

    ocr_images = sum(p.ocr_image_count for p in pages)
    logger.info(
        "Extracted %s pages from %s (%s image regions replaced by OCR text)",
        len(pages),
        Path(file_path).name,
        ocr_images,
    )
    return pages
```

- [ ] **Step 4: 통과 확인**

Run: `cd backend && python -m pytest tests/test_pdf_ocr.py -v`
Expected: 전체 PASS (기존 테스트 포함 회귀 없음)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/pdf_ocr.py backend/tests/test_pdf_ocr.py
git commit -m "feat: persist recognised PDF images as PNG alongside OCR text"
```

---

## Task 3: `DocumentProcessor` — `document_id` 배선과 `image_ids` 승계

**Files:**
- Modify: `backend/app/services/document_processor.py`
- Modify: `backend/app/api/routes/upload.py`
- Modify: `backend/tests/test_document_processor_ocr.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`backend/tests/test_document_processor_ocr.py`에 추가한다. 상단 import에
`from pathlib import Path`를 더한다.

```python
def test_image_ids_are_carried_onto_every_chunk_from_that_page(tmp_path):
    """청크-이미지 연결은 페이지 단위 근사다(설계 문서 3.1절):
    그 페이지에서 나온 모든 청크가 그 페이지의 이미지 전부를 인용한다."""
    from app.services import document_processor as module

    module.settings.image_storage_dir = str(tmp_path / "images")
    processor = DocumentProcessor(ocr=StubOCR("그림 안의 설명 문장"))

    chunks = processor.process_file(
        _pdf_with_image(tmp_path), "diagram.pdf", document_id="doc_test123"
    )

    assert chunks
    assert all(c.metadata["image_ids"] for c in chunks)
    image_id = chunks[0].metadata["image_ids"][0]
    assert (tmp_path / "images" / "doc_test123" / f"{image_id}.png").exists()


def test_no_document_id_skips_image_saving(tmp_path):
    """document_id 없이 호출하는 기존 방식(테스트 등)은 그대로 동작해야 한다."""
    processor = DocumentProcessor(ocr=StubOCR("그림 안의 설명 문장"))

    chunks = processor.process_file(_pdf_with_image(tmp_path), "diagram.pdf")

    assert chunks
    assert all(c.metadata.get("image_ids", []) == [] for c in chunks)
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && python -m pytest tests/test_document_processor_ocr.py -v`
Expected: `TypeError: process_file() got an unexpected keyword argument 'document_id'`

- [ ] **Step 3: `document_processor.py` 수정**

`load_pdf`를 교체한다.

```python
    def load_pdf(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Load a PDF, replacing image regions with their OCR text.

        Falls back to plain text extraction if OCR is disabled or unavailable,
        so an upload never fails just because PaddleOCR could not run.

        Args:
            file_path: Path to the PDF
            filename: Original filename
            document_id: Document ID; required to persist extracted images.
                Without it, text extraction proceeds unchanged and no images
                are saved.

        Returns:
            List of Document objects, one per page
        """
        if not settings.ocr_enabled:
            return PyPDFLoader(file_path).load()

        image_dir = Path(settings.image_storage_dir) / document_id if document_id else None

        try:
            pages: list[PdfPage] = extract_pages(file_path, ocr=self.ocr, image_dir=image_dir)
        except Exception as e:
            logger.warning(
                f"OCR extraction failed for {filename} ({e}); "
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
                        "image_ids": page.image_ids,
                    },
                )
            )

        return documents
```

`load_document`의 시그니처와 PDF 분기를 교체한다.

```python
    def load_document(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Load a document based on file extension.
        """
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == ".pdf":
                documents = self.load_pdf(file_path, filename, document_id)
            elif file_extension == ".txt":
                documents = TextLoader(file_path, encoding='utf-8').load()
            elif file_extension == ".docx":
                documents = Docx2txtLoader(file_path).load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            logger.info(f"Loaded {len(documents)} pages from {filename}")

            for doc in documents:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = file_path

            return documents

        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}")
            raise
```

`process_file`을 교체한다.

```python
    def process_file(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Complete processing pipeline: load and chunk.
        """
        documents = self.load_document(file_path, filename, document_id)
        chunks = self.chunk_documents(documents, filename)

        return chunks
```

- [ ] **Step 4: `upload.py`에서 `document_id` 전달**

`backend/app/api/routes/upload.py`에서 찾는다.

```python
        chunks = doc_processor.process_file(
            str(file_path),
            file.filename
        )
```

교체한다.

```python
        chunks = doc_processor.process_file(
            str(file_path),
            file.filename,
            document_id=document_id
        )
```

- [ ] **Step 5: 통과 확인**

Run: `cd backend && python -m pytest tests/ -v`
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add backend/app/services/document_processor.py backend/app/api/routes/upload.py backend/tests/test_document_processor_ocr.py
git commit -m "feat: thread document_id through the PDF pipeline and carry image_ids onto chunks"
```

---

## Task 4: 이미지 서빙 엔드포인트 및 삭제 시 정리

**Files:**
- Modify: `backend/app/api/routes/documents.py`
- Create: `backend/tests/test_document_images_api.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_document_images_api.py`:

```python
"""이미지 서빙 엔드포인트 테스트.

document_id와 image_id는 사용자 입력이 파일 경로가 되는 지점이다.
경로 이탈 방어가 이 테스트의 핵심이다.
"""
import pytest
from fastapi import HTTPException

from app.api.routes.documents import delete_document_images, resolve_image_path


def test_valid_ids_resolve_inside_storage_root(tmp_path):
    root = tmp_path / "images"
    (root / "doc_abc123").mkdir(parents=True)
    expected = root / "doc_abc123" / "p0_i7.png"
    expected.write_bytes(b"png")

    result = resolve_image_path("doc_abc123", "p0_i7", root)

    assert result == expected.resolve()


@pytest.mark.parametrize("document_id", ["..", "../etc", "doc/../..", "doc abc", "doc;rm"])
def test_path_traversal_in_document_id_is_rejected(document_id, tmp_path):
    with pytest.raises(HTTPException) as exc:
        resolve_image_path(document_id, "p0_i7", tmp_path / "images")

    assert exc.value.status_code == 404


@pytest.mark.parametrize("image_id", ["..", "../../secret", "p0/i7", "p0 i7", "p0.i7"])
def test_path_traversal_in_image_id_is_rejected(image_id, tmp_path):
    with pytest.raises(HTTPException) as exc:
        resolve_image_path("doc_abc123", image_id, tmp_path / "images")

    assert exc.value.status_code == 404


def test_missing_file_is_rejected(tmp_path):
    root = tmp_path / "images"
    (root / "doc_abc123").mkdir(parents=True)

    with pytest.raises(HTTPException) as exc:
        resolve_image_path("doc_abc123", "p9_i99", root)

    assert exc.value.status_code == 404


def test_delete_removes_the_document_image_directory(tmp_path):
    root = tmp_path / "images"
    target = root / "doc_abc123"
    target.mkdir(parents=True)
    (target / "p0_i7.png").write_bytes(b"png")
    other = root / "doc_other"
    other.mkdir()
    (other / "p0_i1.png").write_bytes(b"png")

    delete_document_images("doc_abc123", root)

    assert not target.exists()
    assert other.exists()  # 다른 문서는 건드리지 않는다


def test_delete_with_unsafe_id_is_a_noop(tmp_path):
    root = tmp_path / "images"
    root.mkdir(parents=True)
    (root / "keep.png").write_bytes(b"png")

    delete_document_images("../", root)

    assert (root / "keep.png").exists()


def test_delete_when_directory_absent_does_not_raise(tmp_path):
    delete_document_images("doc_never_existed", tmp_path / "images")
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && python -m pytest tests/test_document_images_api.py -v`
Expected: `ImportError: cannot import name 'resolve_image_path'`

- [ ] **Step 3: `documents.py` 구현**

전체 파일을 아래로 교체한다.

```python
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import re
import shutil
from typing import List

from app.api.models.responses import DocumentInfo
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# document_id와 image_id는 사용자 입력이 파일 경로가 되므로 화이트리스트로 제한한다.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")


def resolve_image_path(document_id: str, image_id: str, storage_root: Path) -> Path:
    """이미지 경로를 안전하게 해석한다.

    화이트리스트 검증과 resolve() 후 루트 하위 확인을 모두 거친다.
    실패 원인을 노출하지 않도록 모든 거부는 404로 통일한다.
    """
    if not _SAFE_ID.match(document_id) or not _SAFE_ID.match(image_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    root = Path(storage_root).resolve()
    candidate = (root / document_id / f"{image_id}.png").resolve()

    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    if not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    return candidate


def delete_document_images(document_id: str, storage_root: Path) -> None:
    """문서의 이미지 디렉터리를 제거한다. 없으면 아무것도 하지 않는다."""
    if not _SAFE_ID.match(document_id):
        logger.warning("Refusing to delete images for unsafe document_id")
        return

    root = Path(storage_root).resolve()
    target = (root / document_id).resolve()

    if not target.is_relative_to(root) or target == root:
        logger.warning("Refusing to delete images outside storage root")
        return

    shutil.rmtree(target, ignore_errors=True)
    logger.info("Removed image directory for document %s", document_id)


@router.get("/{document_id}/images/{image_id}")
async def get_document_image(document_id: str, image_id: str) -> FileResponse:
    """문서에서 추출된 도표 이미지를 서빙한다."""
    path = resolve_image_path(document_id, image_id, Path(settings.image_storage_dir))
    return FileResponse(path, media_type="image/png")


@router.get("/", response_model=List[DocumentInfo])
async def get_documents() -> List[DocumentInfo]:
    """
    Get list of all uploaded documents.

    Note: This is a placeholder implementation.
    In a production system, you would store document metadata in a database.
    """
    # TODO: Implement document listing from database
    logger.info("Get documents endpoint called")
    return []


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks and images.

    Note: This is a placeholder implementation.
    In a production system, you would also delete from the database.
    """
    try:
        from app.services.vector_store import VectorStoreService

        vector_service = VectorStoreService()
        vector_service.delete_by_document_id(document_id)
        delete_document_images(document_id, Path(settings.image_storage_dir))

        return {"message": f"Document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
```

**주의:** `app/main.py`에서 이 라우터가 마운트된 prefix를 확인한다. 위에서
`get_document_image`(`/{document_id}/images/{image_id}`)를 `get_documents`(`/`),
`delete_document`(`/{document_id}`)보다 먼저 등록해 두었다 — 세그먼트 수가 달라
실질적으로 충돌하지는 않지만, FastAPI는 등록 순서대로 매칭을 시도하므로 이 순서를
유지한다.

- [ ] **Step 4: 통과 확인**

Run: `cd backend && python -m pytest tests/test_document_images_api.py -v`
Expected: 8 PASS

- [ ] **Step 5: 커밋**

```bash
git add backend/app/api/routes/documents.py backend/tests/test_document_images_api.py
git commit -m "feat: serve extracted diagram images and clean them up on delete"
```

---

## Task 5: 출처 응답에 `image_urls` 노출

**Files:**
- Modify: `backend/app/api/models/responses.py`
- Modify: `backend/app/services/rag_service.py`
- Create: `backend/tests/test_source_formatting.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_source_formatting.py`:

```python
from langchain.schema import Document

from app.services.rag_service import RAGService


def _format(docs):
    """RAGService를 생성하지 않고 포맷터만 호출한다.

    __init__이 임베딩 모델과 Ollama 연결을 요구하므로 우회한다.
    """
    service = RAGService.__new__(RAGService)
    return service._format_sources(docs)


def test_chunk_with_image_ids_gets_relative_image_urls():
    doc = Document(
        page_content="[이미지 텍스트]\n생활SOC 확충",
        metadata={
            "filename": "고시.pdf", "document_id": "doc_abc123",
            "page": 1, "chunk_index": 7,
            "image_ids": ["p2_a1b2c3", "p2_full"],
        },
    )

    sources = _format([doc])

    assert sources[0].image_urls == [
        "/api/v1/documents/doc_abc123/images/p2_a1b2c3",
        "/api/v1/documents/doc_abc123/images/p2_full",
    ]


def test_chunk_without_image_ids_has_empty_list():
    doc = Document(
        page_content="본문 텍스트",
        metadata={"filename": "고시.pdf", "document_id": "doc_abc123", "chunk_index": 0},
    )

    sources = _format([doc])

    assert sources[0].image_urls == []


def test_missing_document_id_yields_no_urls_even_with_image_ids():
    """document_id가 없으면 깨진 링크를 만들지 않는다."""
    doc = Document(
        page_content="텍스트",
        metadata={"filename": "고시.pdf", "chunk_index": 0, "image_ids": ["p0_x"]},
    )

    sources = _format([doc])

    assert sources[0].image_urls == []
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && python -m pytest tests/test_source_formatting.py -v`
Expected: `AttributeError: 'SourceDocument' object has no attribute 'image_urls'`

- [ ] **Step 3: `SourceDocument`에 필드 추가**

`backend/app/api/models/responses.py`의 `similarity_score` 줄 다음에 추가한다.

```python
    image_urls: list[str] = Field(
        default_factory=list,
        description="이 청크가 속한 페이지에서 발견된 도표 이미지 URL 목록"
    )
```

- [ ] **Step 4: `_format_sources` 교체**

`backend/app/services/rag_service.py`의 `_format_sources`를 교체한다.

```python
    def _format_sources(self, source_docs: list) -> list[SourceDocument]:
        """Format source documents for response."""
        formatted_sources = []

        for doc in source_docs:
            document_id = doc.metadata.get("document_id", "")
            image_ids = doc.metadata.get("image_ids") or []
            image_urls = (
                [f"/api/v1/documents/{document_id}/images/{image_id}" for image_id in image_ids]
                if document_id
                else []
            )

            source = SourceDocument(
                content=doc.page_content,
                document_name=doc.metadata.get("filename", "Unknown"),
                document_id=document_id,
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                similarity_score=None,  # Can add if using similarity_search_with_score
                image_urls=image_urls,
            )
            formatted_sources.append(source)

        return formatted_sources
```

- [ ] **Step 5: 통과 확인**

Run: `cd backend && python -m pytest tests/test_source_formatting.py -v`
Expected: 3 PASS

- [ ] **Step 6: 커밋**

```bash
git add backend/app/api/models/responses.py backend/app/services/rag_service.py backend/tests/test_source_formatting.py
git commit -m "feat: expose diagram image URLs on source citations"
```

---

## Task 6: OCR 출처에 대한 LLM 주의 프롬프트

OCR 텍스트는 이미 `[이미지 텍스트]` 접두사로 본문에 인라인 표시되어 컨텍스트에
그대로 들어간다(설계 문서 3.5절). 빠진 것은 LLM이 그 마커를 어떻게 다뤄야 하는지에
대한 지침뿐이다.

**Files:**
- Modify: `backend/app/services/rag_service.py`
- Modify: `backend/tests/test_local_rag_defaults.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`backend/tests/test_local_rag_defaults.py`에 추가한다.

```python
def test_qa_prompt_warns_about_ocr_marker():
    from app.services.rag_service import QA_PROMPT

    assert "[이미지 텍스트]" in QA_PROMPT.template
    assert "{context}" in QA_PROMPT.template
    assert "{question}" in QA_PROMPT.template
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && python -m pytest tests/test_local_rag_defaults.py -v -k qa_prompt`
Expected: `ImportError: cannot import name 'QA_PROMPT'`

- [ ] **Step 3: `rag_service.py`에 프롬프트 추가**

import 블록에 추가한다.

```python
from langchain.prompts import PromptTemplate
```

모듈 레벨, `RAGService` 클래스 정의 위에 추가한다.

```python
QA_PROMPT = PromptTemplate(
    template="""다음 문맥을 참고해 질문에 답하라.
문맥 중 '[이미지 텍스트]'로 표시된 부분은 문서의 도표·이미지에서 문자 인식(OCR)으로
추출한 것이라 오탈자가 있을 수 있다. 이를 근거로 답할 때는 인명·지명 등 고유명사를
단정하지 말고, 해당 페이지의 도표를 직접 확인하도록 안내하라.

문맥:
{context}

질문: {question}
답변:""",
    input_variables=["context", "question"],
)
```

`ask_question`의 체인 생성부를 교체한다.

```python
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=True
        )
```

- [ ] **Step 4: 통과 확인**

Run: `cd backend && python -m pytest tests/ -v`
Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add backend/app/services/rag_service.py backend/tests/test_local_rag_defaults.py
git commit -m "feat: warn the LLM about OCR-derived text in retrieved context"
```

---

## Task 7: 프런트엔드 — 인라인 썸네일과 라이트박스

**Files:**
- Modify: `frontend/src/types/api.types.ts`
- Modify: `frontend/src/components/SourceCitation.tsx`
- Modify: `frontend/src/styles/SourceCitation.css`

- [ ] **Step 1: 타입에 필드 추가**

`frontend/src/types/api.types.ts`의 `SourceDocument`에 추가한다.

```typescript
export interface SourceDocument {
  content: string;
  document_name: string;
  document_id: string;
  page?: number;
  chunk_index: number;
  similarity_score?: number;
  image_urls: string[];
}
```

- [ ] **Step 2: `SourceCitation.tsx` 교체**

```tsx
import React, { useState } from 'react';
import { FileText, Image as ImageIcon } from 'lucide-react';
import { SourceDocument } from '../types/api.types';
import '../styles/SourceCitation.css';

interface SourceCitationProps {
  source: SourceDocument;
  index: number;
}

export const SourceCitation: React.FC<SourceCitationProps> = ({ source, index }) => {
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const hasImages = source.image_urls.length > 0;

  return (
    <div className="source-citation">
      <div className="source-header">
        <FileText size={16} />
        <span className="source-index">[{index + 1}]</span>
        <span className="source-filename">{source.document_name}</span>
        {source.page !== undefined && (
          <span className="source-page">Page {source.page}</span>
        )}
        {hasImages && (
          <span className="source-badge">
            <ImageIcon size={12} /> 도표
          </span>
        )}
      </div>

      {hasImages && (
        <div className="source-images">
          {source.image_urls.map((url) => (
            <img
              key={url}
              src={url}
              alt="문서 도표"
              loading="lazy"
              className="source-image-thumb"
              onClick={() => setLightboxUrl(url)}
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          ))}
        </div>
      )}

      <div className="source-content">
        <p>{source.content}</p>
      </div>

      {source.similarity_score !== undefined && (
        <div className="source-score">
          Relevance: {(source.similarity_score * 100).toFixed(1)}%
        </div>
      )}

      {lightboxUrl && (
        <div className="source-lightbox" onClick={() => setLightboxUrl(null)}>
          <img src={lightboxUrl} alt="문서 도표 확대" />
        </div>
      )}
    </div>
  );
};
```

이미지 로드 실패 시(`onError`) 썸네일만 숨기고 `source-content`의 OCR 텍스트는
그대로 남는다 — 별도 폴백 분기가 필요 없다. 최초 설계와 달리 OCR 텍스트를 접어두지
않는 이유: 현재 파이프라인에서는 OCR 텍스트가 본문과 한 청크에 섞여 있어(설계 문서
3.1절) 텍스트 전체를 접으면 본문까지 가려진다.

- [ ] **Step 3: 스타일 추가**

`frontend/src/styles/SourceCitation.css`에 추가한다. 기존 클래스명·색상 변수는
파일을 열어 실제 톤에 맞춘다(아래는 최소 골격).

```css
.source-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--accent-bg, #eef2ff);
  color: var(--accent-fg, #4338ca);
}

.source-images {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0;
}

.source-image-thumb {
  max-width: 160px;
  max-height: 120px;
  object-fit: contain;
  border: 1px solid var(--border-color, #e5e7eb);
  border-radius: 6px;
  cursor: zoom-in;
}

.source-lightbox {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  cursor: zoom-out;
}

.source-lightbox img {
  max-width: 90vw;
  max-height: 90vh;
  object-fit: contain;
}
```

- [ ] **Step 4: 수동 확인**

`/run` 스킬로 프런트/백엔드를 띄우고, 도표가 있는 PDF(예: 기존 업로드 샘플)를
업로드한 뒤 도표 관련 질문을 던져 출처 카드에 썸네일이 뜨는지, 클릭 시 라이트박스가
열리는지, 존재하지 않는 image_id에 대해 404가 오는지 확인한다.

- [ ] **Step 5: 커밋**

```bash
git add frontend/src/types/api.types.ts frontend/src/components/SourceCitation.tsx frontend/src/styles/SourceCitation.css
git commit -m "feat: render diagram images inline in source citation cards"
```

---

## 검증 기준

구현 완료의 조건:

1. `pytest tests/ -v` 전체 통과
2. `npx tsc --noEmit` 오류 없음
3. 도표가 있는 PDF 업로드 시 `backend/app/storage/images/doc_*/`에 PNG가 생성됨
4. 도표 관련 질문에 `image_urls`가 비어 있지 않은 출처가 최소 1개 반환됨
5. 해당 URL이 200 + `image/png` 응답
6. 경로 이탈 시도가 404
7. 브라우저 출처 카드에 도표 이미지가 인라인 표시되고 라이트박스가 동작
8. `ocr_enabled=False`로 두면 이 기능 도입 전과 동일하게 동작 (`image_urls`는 항상 빈 리스트)
9. `document_id` 없이 `process_file`을 호출하는 기존 테스트가 그대로 통과

## 리스크 / 후속 확인 사항

- **기존 인덱스와의 호환.** 이 기능 배포 전에 이미 인덱싱된 청크는
  `metadata["image_ids"]`가 없다. `_format_sources`는 `.get("image_ids") or []`로
  방어하므로 깨지지 않는다 — 다만 그 청크들에는 이미지가 뜨지 않으며, 보여주려면
  재업로드가 필요하다.
- **디스크 사용량.** 전체 페이지 스캔 이미지는 `ocr_dpi`(200) 기준 페이지당
  수백 KB 수준으로 예상된다. 대량 업로드 시 `app/storage/images` 증가 추이를
  운영 중 관찰할 필요가 있다 — 이 계획에는 보존 정책(TTL, 용량 상한)이 없다.
- **청크-이미지 연결의 정밀도.** 설계 문서 3.1절에서 명시했듯 페이지 단위 근사라
  관련 없는 이미지가 함께 표시될 수 있다. 사용자 피드백을 보고 필요하면 청크 분할
  경계와 이미지 삽입 위치를 함께 추적하는 정밀 매핑으로 옮겨갈 여지를 남겨둔다.
