"""The OCR side of the process boundary.

PaddleOCR's inference predictor segfaults when torch is loaded in the same
process: `import paddlex` pulls in modelscope, modelscope imports torch, and
Paddle then dies while building the predictor.  The API process loads torch
for the embedding model, so OCR runs here, in a process of its own.

Nothing in this module may import torch, directly or indirectly.
"""

from typing import Any
import sys
import types


def install_modelscope_stub() -> None:
    """Stop ``import paddlex`` from pulling torch in through modelscope.

    Models are fetched from Hugging Face (paddlex's default source), so the
    ModelScope hoster is not needed; a stub keeps torch out of this process.
    """
    existing = sys.modules.get("modelscope")
    if getattr(existing, "__ocr_worker_stub__", False):
        return

    stub = types.ModuleType("modelscope")
    stub.__ocr_worker_stub__ = True

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "ModelScope is disabled in the OCR process; "
            "models are downloaded from Hugging Face instead"
        )

    stub.snapshot_download = _unavailable
    sys.modules["modelscope"] = stub


_service = None


def recognise(image: Any) -> str:
    """Recognise one image. Called in the worker process, once per region."""
    global _service

    if _service is None:
        install_modelscope_stub()

        if "torch" in sys.modules:
            # Building the predictor now would segfault the worker, and the
            # upload would only see it as a dead process. "spawn" re-imports
            # the parent's __main__, so this fires when that module imports
            # torch at top level (outside an `if __name__ == "__main__"`).
            raise RuntimeError(
                "torch is loaded in the OCR worker process, where PaddleOCR "
                "cannot run; the entry point must not import torch at import "
                "time"
            )

        from app.services.ocr_service import PaddleOCRService

        _service = PaddleOCRService()

    return _service.image_to_text(image)
