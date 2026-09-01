"""문서 총 개수 제한(max_documents) 테스트.

업로드 디렉터리를 문서 개수의 근거로 삼는다 - 이 저장소가 이미 다른 곳
(documents.py의 list_parsed_documents 등)에서 쓰는, 저장소 디렉터리를
임시 레지스트리로 취급하는 패턴과 같다.
"""
import pytest
from fastapi import HTTPException

from app.api.routes import upload as upload_module
from app.api.routes.upload import _count_existing_documents


def test_counts_zero_when_dir_missing(tmp_path):
    assert _count_existing_documents(tmp_path / "missing") == 0


def test_counts_distinct_documents(tmp_path):
    (tmp_path / "doc_abcdef012345_a.pdf").write_bytes(b"x")
    (tmp_path / "doc_112233445566_b.txt").write_bytes(b"x")

    assert _count_existing_documents(tmp_path) == 2


def test_ignores_non_matching_files(tmp_path):
    (tmp_path / ".gitkeep").write_bytes(b"")
    (tmp_path / "doc_abcdef012345_a.pdf").write_bytes(b"x")

    assert _count_existing_documents(tmp_path) == 1


def test_ignores_subdirectories(tmp_path):
    (tmp_path / "doc_abcdef012345_a.pdf").write_bytes(b"x")
    (tmp_path / "doc_112233445566_subdir").mkdir()

    assert _count_existing_documents(tmp_path) == 1


def test_enforce_document_cap_raises_when_at_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_module.settings, "max_documents", 2)
    (tmp_path / "doc_abcdef012345_a.pdf").write_bytes(b"x")
    (tmp_path / "doc_112233445566_b.pdf").write_bytes(b"x")

    with pytest.raises(HTTPException) as exc:
        upload_module._enforce_document_cap(tmp_path)

    assert exc.value.status_code == 400
    assert "Maximum of 2" in exc.value.detail


def test_enforce_document_cap_allows_when_under_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_module.settings, "max_documents", 2)
    (tmp_path / "doc_abcdef012345_a.pdf").write_bytes(b"x")

    upload_module._enforce_document_cap(tmp_path)  # must not raise


def test_enforce_document_cap_allows_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_module.settings, "max_documents", 2)

    upload_module._enforce_document_cap(tmp_path / "missing")  # must not raise
