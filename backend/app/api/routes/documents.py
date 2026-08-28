from fastapi import APIRouter, HTTPException, status
import logging
from typing import List

from app.api.models.responses import DocumentInfo

logger = logging.getLogger(__name__)
router = APIRouter()


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
    Delete a document and all its associated chunks.

    Note: This is a placeholder implementation.
    In a production system, you would also delete from the database.
    """
    try:
        from app.services.vector_store import VectorStoreService

        vector_service = VectorStoreService()
        vector_service.delete_by_document_id(document_id)

        return {"message": f"Document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
