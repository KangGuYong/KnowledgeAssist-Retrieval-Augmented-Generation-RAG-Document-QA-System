from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import re
import shutil
from typing import List

from app.api.models.responses import DocumentInfo
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# document_id와 image_id는 사용자 입력이 파일 경로가 되므로 화이트리스트로 제한한다.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")


def resolve_image_path(document_id: str, image_id: str, storage_root: Path) -> Path:
    """이미지 경로를 안전하게 해석한다.

    화이트리스트 검증과 resolve() 후 루트 하위 확인을 모두 거친다.
    실패 원인을 노출하지 않도록 모든 거부는 404로 통일한다.
    """
    if not _SAFE_ID.match(document_id) or not _SAFE_ID.match(image_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    root = Path(storage_root).resolve()
    candidate = (root / document_id / f"{image_id}.png").resolve()

    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    if not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    return candidate


def delete_document_images(document_id: str, storage_root: Path) -> None:
    """문서의 이미지 디렉터리를 제거한다. 없으면 아무것도 하지 않는다."""
    if not _SAFE_ID.match(document_id):
        logger.warning("Refusing to delete images for unsafe document_id")
        return

    root = Path(storage_root).resolve()
    target = (root / document_id).resolve()

    if not target.is_relative_to(root) or target == root:
        logger.warning("Refusing to delete images outside storage root")
        return

    shutil.rmtree(target, ignore_errors=True)
    logger.info("Removed image directory for document %s", document_id)


@router.get("/{document_id}/images/{image_id}")
async def get_document_image(document_id: str, image_id: str) -> FileResponse:
    """문서에서 추출된 도표 이미지를 서빙한다."""
    path = resolve_image_path(document_id, image_id, Path(settings.image_storage_dir))
    return FileResponse(path, media_type="image/png")


@router.get("/", response_model=List[DocumentInfo])
async def get_documents() -> List[DocumentInfo]:
    """
    Get list of all uploaded documents.

    Note: This is a placeholder implementation.
    In a production system, you would store document metadata in a database.
    """
    # TODO: Implement document listing from database
    logger.info("Get documents endpoint called")
    return []


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks and images.

    Note: This is a placeholder implementation.
    In a production system, you would also delete from the database.
    """
    try:
        from app.services.vector_store import VectorStoreService

        vector_service = VectorStoreService()
        vector_service.delete_by_document_id(document_id)
        delete_document_images(document_id, Path(settings.image_storage_dir))

        return {"message": f"Document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
