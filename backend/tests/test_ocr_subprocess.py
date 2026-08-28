"""OCR must run outside the API process, and survive a worker dying."""

import os
import sys
import time

from app.services import ocr_service, ocr_worker
from app.services.ocr_service import PaddleOCRService, SubprocessOCRService

from tests.ocr_targets import crash, echo_pid, hang, import_torch_then_recognise


def test_recognition_happens_in_another_process():
    service = SubprocessOCRService(target=echo_pid)
    try:
        result = service.image_to_text("payload")
    finally:
        service.shutdown()

    assert "image=payload" in result
    assert f"pid={os.getpid()}" not in result


def test_the_worker_is_reused_across_calls():
    service = SubprocessOCRService(target=echo_pid)
    try:
        first = service.image_to_text("a")
        second = service.image_to_text("b")
    finally:
        service.shutdown()

    assert first.split()[0] == second.split()[0]  # same worker pid


def test_a_dead_worker_is_replaced_and_the_upload_survives():
    service = SubprocessOCRService(target=crash)
    try:
        assert service.image_to_text("boom") == ""

        service._target = echo_pid  # next page recognises normally again
        assert "image=ok" in service.image_to_text("ok")
    finally:
        service.shutdown()


def test_a_stuck_worker_times_out_instead_of_blocking_the_upload():
    service = SubprocessOCRService(target=hang, timeout=2)
    started = time.monotonic()
    try:
        assert service.image_to_text("slow") == ""
    finally:
        service.shutdown()

    assert time.monotonic() - started < 60


def test_isolation_is_the_default_and_can_be_turned_off(monkeypatch):
    ocr_service.get_ocr_service.cache_clear()
    assert isinstance(ocr_service.get_ocr_service(), SubprocessOCRService)

    ocr_service.get_ocr_service.cache_clear()
    monkeypatch.setattr(ocr_service.settings, "ocr_isolate_process", False)
    assert isinstance(ocr_service.get_ocr_service(), PaddleOCRService)
    ocr_service.get_ocr_service.cache_clear()


def test_the_worker_keeps_torch_out_of_its_process(monkeypatch):
    """paddlex imports modelscope, which imports torch: stub it away first."""
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)

    ocr_worker.install_modelscope_stub()
    import modelscope

    assert modelscope.__ocr_worker_stub__ is True
    try:
        modelscope.snapshot_download("anything")
    except RuntimeError as e:
        assert "Hugging Face" in str(e)
    else:
        raise AssertionError("the stub must not silently download")

    monkeypatch.delitem(sys.modules, "modelscope", raising=False)


def test_a_worker_polluted_with_torch_reports_why_instead_of_segfaulting():
    service = SubprocessOCRService(target=import_torch_then_recognise, timeout=60)
    try:
        assert service.image_to_text("payload") == ""
    finally:
        service.shutdown()
