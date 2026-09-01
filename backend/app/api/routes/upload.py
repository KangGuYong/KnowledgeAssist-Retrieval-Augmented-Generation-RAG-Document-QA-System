from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pathlib import Path
import aiofiles
import re
import uuid
import logging
from typing import List, Optional

from app.config import get_settings
from app.api.models.responses import UploadResponse
from app.services.chunking import VALID_STRATEGIES
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# Initialize services
doc_processor = DocumentProcessor()
vector_service = VectorStoreService()

_DOCUMENT_ID_PREFIX = re.compile(r"^(doc_[0-9a-f]{12})_")


def _count_existing_documents(upload_dir: Path) -> int:
    """Count distinct documents currently in the system.

    Uses upload_dir as the source of truth (one file per document, named
    "{document_id}_{filename}") rather than querying Chroma or the
    parsed-result store - this codebase already treats storage directories
    as ad-hoc registries elsewhere (see documents.py's list_parsed_documents),
    and upload_dir is the only one guaranteed to have an entry for every
    successfully uploaded document regardless of file type or parse status.
    """
    if not upload_dir.is_dir():
        return 0

    document_ids = set()
    for path in upload_dir.iterdir():
        if not path.is_file():
            continue
        match = _DOCUMENT_ID_PREFIX.match(path.name)
        if match:
            document_ids.add(match.group(1))

    return len(document_ids)


def _enforce_document_cap(upload_dir: Path) -> None:
    """Raise 400 if the system is already at settings.max_documents.

    Called before any file I/O in _process_upload, so a rejected upload
    never touches disk.
    """
    if _count_existing_documents(upload_dir) >= settings.max_documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Maximum of {settings.max_documents} documents already uploaded. "
                "Delete an existing document before uploading a new one."
            ),
        )


async def _process_upload(
    file: UploadFile,
    chunking_strategy: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
) -> UploadResponse:
    """
    Upload and process a single document.

    This is the shared logic behind both `/` and `/batch` - kept as a plain
    function (not the route handler itself) because upload_multiple_files
    calls it directly per file, and FastAPI's Form(...) defaults only
    resolve to real values through the dependency-injection path, not when
    called as an ordinary Python function.

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

    # Enforce the total-document cap before doing any real work
    _enforce_document_cap(Path(settings.upload_dir))

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
            file.filename,
            document_id=document_id,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Add to vector store
        vector_service.add_documents(chunks, document_id)

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=len(chunks),
            status="processed",
            message=f"Successfully processed {file.filename}",
            chunking_strategy=chunking_strategy,
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


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(settings.chunking_strategy),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
) -> UploadResponse:
    """Upload and process a single document."""
    if chunking_strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown chunking_strategy: {chunking_strategy}. Allowed: {sorted(VALID_STRATEGIES)}"
        )

    return await _process_upload(file, chunking_strategy, chunk_size, chunk_overlap)


@router.post("/batch", response_model=List[UploadResponse])
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    chunking_strategy: str = Form(settings.chunking_strategy),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
) -> List[UploadResponse]:
    """
    Upload and process multiple documents.

    chunking_strategy/chunk_size/chunk_overlap are shared across every file
    in the batch - to use different settings per file, upload them one at a
    time via `/`.
    """
    if chunking_strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown chunking_strategy: {chunking_strategy}. Allowed: {sorted(VALID_STRATEGIES)}"
        )

    responses = []

    for file in files:
        try:
            response = await _process_upload(file, chunking_strategy, chunk_size, chunk_overlap)
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
                    message=e.detail,
                    chunking_strategy=chunking_strategy,
                )
            )

    return responses
