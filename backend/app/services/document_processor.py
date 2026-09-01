from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from pathlib import Path
import logging
import time
from typing import Optional

from app.config import get_settings
from app.services.chunking import build_splitter
from app.services.mineru_client import PdfPage, parse_and_build_pages
from app.services import parsed_store

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Process and chunk documents."""

    def __init__(self, ocr=None, mineru_client=None):
        # Defaults to the shared PaddleOCR service; injectable for tests.
        self.ocr = ocr
        # Defaults to a real MineruClient(); injectable for tests.
        self.mineru_client = mineru_client

    def load_pdf(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Load a PDF via MinerU, augmenting image blocks with PaddleOCR text.

        Falls back to plain text extraction if MinerU is disabled or
        unavailable, so an upload never fails just because the parsing
        service could not run.

        Args:
            file_path: Path to the PDF
            filename: Original filename
            document_id: Document ID; required to persist extracted images
                and the parsed-result JSON (see parsed_store.save). Without
                it, text extraction proceeds unchanged and neither is saved.

        Returns:
            List of Document objects, one per page
        """
        if not settings.mineru_enabled:
            return PyPDFLoader(file_path).load()

        image_dir = Path(settings.image_storage_dir) / document_id if document_id else None

        ocr = None
        if settings.ocr_enabled:
            ocr = self.ocr
            if ocr is None:
                from app.services.ocr_service import get_ocr_service

                ocr = get_ocr_service()

        on_parsed = None
        if document_id:
            on_parsed = lambda result: parsed_store.save(document_id, filename, result, image_dir)

        try:
            pages: list[PdfPage] = parse_and_build_pages(
                file_path, ocr=ocr, image_dir=image_dir, client=self.mineru_client, on_parsed=on_parsed
            )
        except Exception as e:
            logger.warning(
                f"MinerU extraction failed for {filename} ({e}); "
                "falling back to text-only extraction"
            )
            return PyPDFLoader(file_path).load()

        documents = []
        for page in pages:
            if not page.text.strip():
                continue
            documents.append(
                Document(
                    page_content=page.text,
                    metadata={
                        "page": page.page_number - 1,  # 0-based, as PyPDFLoader
                        "page_number": page.page_number,
                        "ocr_used": page.ocr_image_count > 0,
                        "ocr_image_count": page.ocr_image_count,
                        "full_page_ocr": page.full_page_ocr,
                        # Chroma metadata values must be str/int/float/bool, so a
                        # list can't be stored directly; image ids never contain
                        # commas (md5 hexdigest), so joining is lossless.
                        "image_ids": ",".join(page.image_ids),
                    },
                )
            )

        return documents

    def load_document(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Load a document based on file extension.

        Args:
            file_path: Path to the file
            filename: Original filename
            document_id: Document ID; forwarded to load_pdf to scope saved images.

        Returns:
            List of Document objects
        """
        file_extension = Path(filename).suffix.lower()
        start = time.perf_counter()

        try:
            if file_extension == ".pdf":
                documents = self.load_pdf(file_path, filename, document_id)
            elif file_extension == ".txt":
                documents = TextLoader(file_path, encoding='utf-8').load()
            elif file_extension == ".docx":
                documents = Docx2txtLoader(file_path).load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            logger.info(
                "[TIMING] Document load (%s): %.2fs, %d page(s) - %s",
                file_extension, time.perf_counter() - start, len(documents), filename,
            )

            # Add filename to metadata
            for doc in documents:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = file_path

            return documents

        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}")
            raise

    def chunk_documents(
        self,
        documents: list[Document],
        filename: str,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects
            filename: Original filename for metadata
            chunking_strategy: "default" or "semantic"; falls back to
                settings.chunking_strategy when not given.
            chunk_size: Only used by "default"; falls back to settings.chunk_size.
            chunk_overlap: Only used by "default"; falls back to settings.chunk_overlap.

        Returns:
            List of chunked Document objects
        """
        strategy = chunking_strategy or settings.chunking_strategy
        resolved_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        resolved_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        start = time.perf_counter()
        splitter = build_splitter(strategy, resolved_chunk_size, resolved_chunk_overlap)
        chunks = splitter.split_documents(documents)
        elapsed = time.perf_counter() - start

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["filename"] = filename
            chunk.metadata["chunking_strategy"] = strategy
            # Chroma metadata must be scalar (str/int/float/bool); None is
            # rejected outright (same failure mode already hit with image_ids,
            # see commit af38b9f). chunk_size/overlap only apply to "default",
            # so omit the keys for "semantic" rather than writing None.
            if strategy == "default":
                chunk.metadata["chunk_size"] = resolved_chunk_size
                chunk.metadata["chunk_overlap"] = resolved_chunk_overlap

        logger.info(
            "[TIMING] Chunking (%s): %.2fs, %d chunk(s) - %s",
            strategy, elapsed, len(chunks), filename,
        )
        return chunks

    def process_file(
        self,
        file_path: str,
        filename: str,
        document_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[Document]:
        """
        Complete processing pipeline: load and chunk.

        Args:
            file_path: Path to the file
            filename: Original filename
            document_id: Document ID; forwarded to load_document to scope saved images.
            chunking_strategy: "default" or "semantic"; forwarded to chunk_documents.
            chunk_size: Only used by "default"; forwarded to chunk_documents.
            chunk_overlap: Only used by "default"; forwarded to chunk_documents.

        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_document(file_path, filename, document_id)
        chunks = self.chunk_documents(
            documents, filename, chunking_strategy, chunk_size, chunk_overlap
        )

        return chunks
