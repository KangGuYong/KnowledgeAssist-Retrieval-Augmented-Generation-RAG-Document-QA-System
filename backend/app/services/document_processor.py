from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from pathlib import Path
import logging
from typing import Optional

from app.config import get_settings
from app.services.pdf_ocr import PdfPage, extract_pages

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Process and chunk documents."""

    def __init__(self, ocr=None):
        # Defaults to the shared PaddleOCR service; injectable for tests.
        self.ocr = ocr
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf(self, file_path: str, filename: str) -> list[Document]:
        """
        Load a PDF, replacing image regions with their OCR text.

        Falls back to plain text extraction if OCR is disabled or unavailable,
        so an upload never fails just because PaddleOCR could not run.

        Args:
            file_path: Path to the PDF
            filename: Original filename

        Returns:
            List of Document objects, one per page
        """
        if not settings.ocr_enabled:
            return PyPDFLoader(file_path).load()

        try:
            pages: list[PdfPage] = extract_pages(file_path, ocr=self.ocr)
        except Exception as e:
            logger.warning(
                f"OCR extraction failed for {filename} ({e}); "
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
                    },
                )
            )

        return documents

    def load_document(self, file_path: str, filename: str) -> list[Document]:
        """
        Load a document based on file extension.

        Args:
            file_path: Path to the file
            filename: Original filename

        Returns:
            List of Document objects
        """
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == ".pdf":
                documents = self.load_pdf(file_path, filename)
            elif file_extension == ".txt":
                documents = TextLoader(file_path, encoding='utf-8').load()
            elif file_extension == ".docx":
                documents = Docx2txtLoader(file_path).load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            logger.info(f"Loaded {len(documents)} pages from {filename}")

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
        filename: str
    ) -> list[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects
            filename: Original filename for metadata

        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index to metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["filename"] = filename

        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks

    def process_file(
        self,
        file_path: str,
        filename: str
    ) -> list[Document]:
        """
        Complete processing pipeline: load and chunk.

        Args:
            file_path: Path to the file
            filename: Original filename

        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_document(file_path, filename)
        chunks = self.chunk_documents(documents, filename)

        return chunks
