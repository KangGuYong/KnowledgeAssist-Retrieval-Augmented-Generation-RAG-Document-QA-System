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


def test_load_parsed_document_returns_404_not_500_for_corrupt_json(tmp_path):
    """A truncated/corrupt parsed JSON file (e.g. left by a crash mid-write,
    before the atomic-write fix) must look like a missing file to the
    client, not surface as an unhandled 500."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "doc_corrupt.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        load_parsed_document("doc_corrupt", tmp_path)

    assert exc.value.status_code == 404


def test_load_parsed_document_returns_404_for_json_missing_required_fields(tmp_path):
    """Valid JSON that doesn't match ParsedDocumentDetail's schema (e.g. a
    pydantic validation error) must also come back as 404, not 500."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "doc_bad_shape.json").write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        load_parsed_document("doc_bad_shape", tmp_path)

    assert exc.value.status_code == 404
