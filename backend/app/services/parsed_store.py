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
