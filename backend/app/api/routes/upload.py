from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pathlib import Path
import aiofiles
import uuid
import logging
from typing import List

from app.config import get_settings
from app.api.models.responses import UploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# Initialize services
doc_processor = DocumentProcessor()
vector_service = VectorStoreService()


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...)
) -> UploadResponse:
    """
    Upload and process a single document.

    This endpoint:
    1. Validates the file type and size
    2. Saves the file temporarily
    3. Loads and chunks the document
    4. Generates embeddings and stores in vector DB
    5. Returns document metadata
    """
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported. Allowed: {settings.allowed_extensions}"
        )

    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.max_upload_size} bytes"
        )

    # Generate unique document ID
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    # Save file temporarily
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{document_id}_{file.filename}"

    try:
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"Saved file {file.filename} to {file_path}")

        # Process document (load and chunk)
        chunks = doc_processor.process_file(
            str(file_path),
            file.filename
        )

        # Add to vector store
        vector_service.add_documents(chunks, document_id)

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=len(chunks),
            status="processed",
            message=f"Successfully processed {file.filename}"
        )

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.post("/batch", response_model=List[UploadResponse])
async def upload_multiple_files(
    files: List[UploadFile] = File(...)
) -> List[UploadResponse]:
    """
    Upload and process multiple documents.
    """
    responses = []

    for file in files:
        try:
            response = await upload_file(file)
            responses.append(response)
        except HTTPException as e:
            # Continue processing other files even if one fails
            logger.error(f"Failed to process {file.filename}: {e.detail}")
            responses.append(
                UploadResponse(
                    document_id="",
                    filename=file.filename,
                    num_chunks=0,
                    status="failed",
                    message=e.detail
                )
            )

    return responses
