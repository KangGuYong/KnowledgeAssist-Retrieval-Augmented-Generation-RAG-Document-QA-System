"""PDF text extraction that replaces image regions with their OCR text.

Each page is decomposed into its layout blocks with PyMuPDF.  Text blocks are
kept as-is, image blocks are rendered and handed to PaddleOCR, and the
recognised text is spliced back in at the position the image occupied.  The
resulting page text is plain text, ready for the normal chunk/embed pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol
import hashlib
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

TEXT_BLOCK = 0
IMAGE_BLOCK = 1


class SupportsImageOcr(Protocol):
    """Minimal interface required from an OCR backend."""

    def image_to_text(self, image: Any) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class PdfPage:
    """One extracted page, with image regions already replaced by text."""

    page_number: int  # 1-based
    text: str
    image_count: int = 0
    ocr_image_count: int = 0
    full_page_ocr: bool = False
    image_ids: list[str] = field(default_factory=list)


@dataclass
class _Block:
    order: tuple[float, float]
    text: str


@dataclass
class _OcrStats:
    images: int = 0
    ocr_images: int = 0
    image_ids: list[str] = field(default_factory=list)


def _pixmap_to_array(pixmap) -> Any:
    """Convert a PyMuPDF pixmap into a BGR numpy array for PaddleOCR."""
    import numpy as np

    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _render(page, clip=None, dpi: Optional[int] = None) -> Any:
    """Render a page (or a clipped region of it) as a BGR numpy array."""
    import fitz

    pixmap = page.get_pixmap(
        clip=clip,
        dpi=dpi or settings.ocr_dpi,
        colorspace=fitz.csRGB,
        alpha=False,
    )
    return _pixmap_to_array(pixmap)


def _text_from_block(block: dict) -> str:
    """Flatten a PyMuPDF text block into a string."""
    lines = []
    for line in block.get("lines", []):
        spans = "".join(span.get("text", "") for span in line.get("spans", []))
        if spans.strip():
            lines.append(spans.strip())
    return "\n".join(lines)


def _format_ocr_block(text: str) -> str:
    """Wrap OCR output so retrieved chunks show where the text came from."""
    prefix = settings.ocr_block_prefix.strip()
    if not prefix:
        return text
    return f"{prefix}\n{text}"


def _is_image_too_small(bbox) -> bool:
    width = abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    return min(width, height) < settings.ocr_min_image_size


def _save_image(image: Any, image_dir: Path, image_id: str) -> bool:
    """Persist a rendered BGR array as PNG. Never raises; logs and returns False on failure.

    The caller must only surface image_id (in PdfPage.image_ids) when this
    returns True — otherwise a citation would point at a PNG that was never
    written.
    """
    try:
        from PIL import Image as PILImage

        image_dir.mkdir(parents=True, exist_ok=True)
        rgb = image[:, :, ::-1]  # BGR -> RGB, matches _pixmap_to_array's channel order
        PILImage.fromarray(rgb).save(image_dir / f"{image_id}.png", format="PNG")
        return True
    except Exception as e:
        logger.warning("Failed to save extracted image %s: %s", image_id, e)
        return False


def _ocr_image_block(
    page,
    block: dict,
    ocr: SupportsImageOcr,
    cache: dict[str, tuple[str, str]],
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

    image_id = None
    if image_dir is not None and key is not None and _save_image(image, image_dir, key):
        image_id = key

    result = (text, image_id)
    if key is not None:
        cache[key] = result
    return result


def _blocks_in_reading_order(blocks: list[_Block]) -> list[_Block]:
    if settings.ocr_layout_order == "native":
        return blocks
    tolerance = max(settings.ocr_row_tolerance, 1.0)
    return sorted(blocks, key=lambda b: (round(b.order[1] / tolerance), b.order[0]))


def _extract_page(
    page,
    page_number: int,
    ocr: SupportsImageOcr,
    cache: Optional[dict[str, tuple[str, str]]] = None,
    image_dir: Optional[Path] = None,
) -> PdfPage:
    """Extract one page, replacing every image region with its OCR text.

    ``cache`` maps image content to already recognised (text, image_id) pairs,
    so a logo or a header image repeated across pages is only read and saved
    once.
    """
    stats = _OcrStats()
    cache = {} if cache is None else cache
    layout = page.get_text("dict")
    raw_blocks = layout.get("blocks", [])

    text_blocks: list[_Block] = []
    image_blocks: list[dict] = []

    for block in raw_blocks:
        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        order = (float(bbox[0]), float(bbox[1]))

        if block.get("type") == TEXT_BLOCK:
            text = _text_from_block(block)
            if text:
                text_blocks.append(_Block(order=order, text=text))
        elif block.get("type") == IMAGE_BLOCK:
            if _is_image_too_small(bbox):
                continue
            image_blocks.append(block)

    native_text = "\n\n".join(b.text for b in text_blocks)
    too_many_images = len(image_blocks) > settings.ocr_max_images_per_page
    looks_scanned = (
        bool(image_blocks) and len(native_text) < settings.ocr_page_text_threshold
    )

    # A scanned page (or a page sliced into many image fragments) is cheaper and
    # more accurate to recognise in a single pass: the rendered page already
    # contains whatever native text it has.
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
        saved = (
            image_dir is not None
            and rendered is not None
            and _save_image(rendered, image_dir, image_id)
        )
        return PdfPage(
            page_number=page_number,
            text=_format_ocr_block(text),
            image_count=len(image_blocks),
            ocr_image_count=len(image_blocks),
            full_page_ocr=True,
            image_ids=[image_id] if saved else [],
        )

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
