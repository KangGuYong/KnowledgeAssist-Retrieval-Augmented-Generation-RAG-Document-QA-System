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
