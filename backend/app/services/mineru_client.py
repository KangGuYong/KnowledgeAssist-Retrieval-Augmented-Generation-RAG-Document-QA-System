"""HTTP client for the self-hosted MinerU document-parsing service.

MinerU replaces this app's PyMuPDF-based PDF layout analysis - see
docs/superpowers/specs/2026-08-31-mineru-pdf-parsing-design.md. This module
only talks to the service and returns its content_list blocks plus their
images (as base64 data URIs - MinerU 3.4.5's /file_parse embeds images
inline rather than exposing a shared output directory, verified 2026-08-31).
Page-grouping and text assembly live in build_pages() below.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
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


def _lookup_image(images: dict, img_path: str) -> str:
    """Resolve a content_list block's img_path to its data URI in `images`.

    Verified against a live MinerU 3.4.5 /file_parse response (2026-08-31):
    content_list blocks' img_path includes a directory prefix (e.g.
    "images/<hash>.jpg"), but the images dict is actually keyed by the bare
    filename (e.g. "<hash>.jpg") - they are NOT the same string. Try the
    exact key first (forward-compatible if a future version matches
    directly), then fall back to matching by basename.
    """
    if img_path in images:
        return images[img_path]
    return images[Path(img_path).name]


def _bgr_array_from_bytes(raw: bytes) -> Any:
    """Decode image bytes into a BGR numpy array (matches PaddleOCR's expected order)."""
    import io

    import numpy as np
    from PIL import Image as PILImage

    rgb = np.array(PILImage.open(io.BytesIO(raw)).convert("RGB"))
    return rgb[:, :, ::-1]


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
    raw = _decode_data_uri(_lookup_image(images, block["img_path"]))
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


def _needs_ocr(block: dict) -> bool:
    """True when a block has img_path but no usable structured text of its
    own (table_body / equation text) - it needs PaddleOCR augmentation
    rather than its own recognised content. A table or equation block whose
    own recognition failed server-side (empty table_body/text) falls back
    here instead of being silently dropped, since it still carries an
    img_path screenshot per MinerU's schema.
    """
    if (block.get("table_body") or "").strip():
        return False
    if block.get("type") == "equation" and (block.get("text") or "").strip():
        return False
    return bool(block.get("img_path"))


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
        by_page.setdefault(block.get("page_idx", 0), []).append(block)

    cache: dict = {}
    pages = []
    for page_idx in sorted(by_page):
        page_blocks = by_page[page_idx]
        parts = []
        ocr_image_count = 0
        image_ids = []

        for block in page_blocks:
            try:
                if _needs_ocr(block):
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
            except Exception as e:
                logger.warning(
                    "Failed to process a content block on page %s: %s", page_idx + 1, e
                )
                continue

        pages.append(
            PdfPage(
                page_number=page_idx + 1,
                text="\n\n".join(parts),
                image_count=sum(1 for b in page_blocks if _needs_ocr(b)),
                ocr_image_count=ocr_image_count,
                image_ids=image_ids,
            )
        )

    return pages


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
