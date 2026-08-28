"""Picklable stand-ins for the OCR worker, used by the subprocess tests."""

import os
import time


def echo_pid(image) -> str:
    """Report the process the call ran in, so isolation can be asserted."""
    return f"pid={os.getpid()} image={image}"


def crash(image) -> str:
    os._exit(1)


def hang(image) -> str:
    time.sleep(120)
    return "too late"


def import_torch_then_recognise(image) -> str:
    """Simulate a worker whose parent __main__ dragged torch in."""
    import sys
    import types

    sys.modules.setdefault("torch", types.ModuleType("torch"))
    from app.services import ocr_worker

    return ocr_worker.recognise(image)
