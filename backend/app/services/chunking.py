"""Text chunking strategies: character-based (default) and semantic.

"default" splits on a fixed character budget (RecursiveCharacterTextSplitter).
"semantic" groups sentences by embedding-similarity breakpoints (LangChain's
SemanticChunker), reusing the app's KURE-v1 embeddings so no second model is
loaded.
"""

from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import BaseDocumentTransformer
from langchain_core.embeddings import Embeddings

from app.config import get_settings

settings = get_settings()

VALID_STRATEGIES = {"default", "semantic"}


def build_splitter(
    strategy: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embeddings: Optional[Embeddings] = None,
) -> BaseDocumentTransformer:
    """Build a text splitter for the given strategy.

    Args:
        strategy: "default" or "semantic".
        chunk_size: Only used by "default"; falls back to settings.chunk_size.
        chunk_overlap: Only used by "default"; falls back to settings.chunk_overlap.
        embeddings: Only used by "semantic"; defaults to the shared KURE-v1
            embeddings. Tests inject a fake here to avoid loading the real model.

    Raises:
        ValueError: strategy is not one of VALID_STRATEGIES.
    """
    if strategy == "default":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        from app.services.vector_store import get_embeddings

        return SemanticChunker(
            embeddings if embeddings is not None else get_embeddings(),
            breakpoint_threshold_type=settings.semantic_chunker_breakpoint_type,
        )

    raise ValueError(f"Unknown chunking strategy: {strategy}")
