from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import Optional
import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache()
def get_embeddings():
    """Get cached embedding model."""
    logger.info(f"Loading embedding model: {settings.embedding_model} (device={settings.embedding_device})")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={'device': settings.embedding_device},
        encode_kwargs={'normalize_embeddings': True}
    )


@lru_cache()
def get_vector_store() -> Chroma:
    """Get or create ChromaDB vector store."""
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir
    )

    logger.info("Vector store initialized")
    return vector_store


class VectorStoreService:
    """Service for vector store operations."""

    def __init__(self):
        self.vector_store = get_vector_store()

    def add_documents(
        self,
        documents: list[Document],
        document_id: str
    ) -> list[str]:
        """
        Add documents to vector store.

        Args:
            documents: List of LangChain Document objects
            document_id: Unique identifier for the source document

        Returns:
            List of chunk IDs
        """
        # Add document_id to metadata
        for doc in documents:
            doc.metadata["document_id"] = document_id

        ids = self.vector_store.add_documents(documents)
        logger.info(f"Added {len(ids)} chunks for document {document_id}")

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> list[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"document_id": "doc_123"})

        Returns:
            List of relevant Document objects
        """
        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores."""
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )

        return results

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks for a specific document.

        Chroma.delete() forwards **kwargs straight to the underlying chromadb
        collection, so `where=` works. It did not in langchain-community
        0.0.38, which passed only `ids=` and dropped the rest - hence the
        `_collection` call this replaces.
        """
        self.vector_store.delete(where={"document_id": document_id})
        logger.info(f"Deleted chunks for document {document_id}")
