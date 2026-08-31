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
