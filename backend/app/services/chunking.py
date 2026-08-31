"""Text chunking strategies: character-based (default) and semantic.

"default" splits on a fixed character budget (RecursiveCharacterTextSplitter).
"semantic" groups sentences by embedding-similarity breakpoints, ported
directly from Greg Kamradt's "5 Levels of Text Splitting" notebook (Level 4:
Semantic Chunking) -
https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
- rather than langchain_experimental's SemanticChunker, so the algorithm is
owned directly instead of living behind a third-party "experimental" package.
Reuses the app's KURE-v1 embeddings so no second model is loaded.
"""

import re
from typing import Any, List, Optional, Protocol

import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from app.config import get_settings

settings = get_settings()

VALID_STRATEGIES = {"default", "semantic"}

# Matches the notebook: split on '.', '?', '!' followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


class DocumentSplitter(Protocol):
    """Minimal interface chunk_documents() relies on - either splitter works."""

    def split_documents(self, documents: List[Document]) -> List[Document]: ...


def _combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    """Combine each sentence with its neighbours into one string, per sentence.

    Reduces noise in the per-sentence embedding: comparing single short
    sentences is noisy, comparing a window of buffer_size neighbours on each
    side is more stable (notebook cell 84).
    """
    for i in range(len(sentences)):
        combined = ""
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined += sentences[j]["sentence"] + " "
        combined += sentences[i]["sentence"]
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined += " " + sentences[j]["sentence"]
        sentences[i]["combined_sentence"] = combined
    return sentences


def _cosine_distances(embeddings: List[List[float]]) -> List[float]:
    """Cosine distance (1 - cosine similarity) between consecutive embeddings.

    Plain numpy instead of sklearn's cosine_similarity (notebook cell 93) to
    avoid adding scikit-learn as a dependency for one function.
    """
    distances = []
    for i in range(len(embeddings) - 1):
        a = np.asarray(embeddings[i])
        b = np.asarray(embeddings[i + 1])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        similarity = float(np.dot(a, b) / denom) if denom else 0.0
        distances.append(1 - similarity)
    return distances


class SemanticChunker:
    """Splits text into chunks at embedding-similarity breakpoints.

    Any distance above the given percentile of all consecutive-sentence
    distances is treated as a breakpoint (notebook cell 102) - the notebook's
    only tunable parameter.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        breakpoint_percentile: float = 95.0,
    ):
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_percentile = breakpoint_percentile

    def split_text(self, text: str) -> List[str]:
        single_sentences = _SENTENCE_SPLIT_RE.split(text)
        if len(single_sentences) <= 1:
            return single_sentences

        sentences = [{"sentence": s, "index": i} for i, s in enumerate(single_sentences)]
        sentences = _combine_sentences(sentences, self.buffer_size)

        embeddings = self.embeddings.embed_documents(
            [s["combined_sentence"] for s in sentences]
        )
        distances = _cosine_distances(embeddings)
        if not distances:
            return [s["sentence"] for s in sentences]

        threshold = np.percentile(distances, self.breakpoint_percentile)
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        chunks = []
        start = 0
        for index in breakpoints:
            chunks.append(" ".join(s["sentence"] for s in sentences[start:index + 1]))
            start = index + 1
        if start < len(sentences):
            chunks.append(" ".join(s["sentence"] for s in sentences[start:]))
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents, carrying each parent's metadata onto every chunk."""
        result = []
        for doc in documents:
            for chunk_text in self.split_text(doc.page_content):
                result.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
        return result


def build_splitter(
    strategy: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embeddings: Optional[Embeddings] = None,
) -> Any:
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
        from app.services.vector_store import get_embeddings

        return SemanticChunker(
            embeddings if embeddings is not None else get_embeddings(),
            breakpoint_percentile=settings.semantic_chunker_breakpoint_percentile,
        )

    raise ValueError(f"Unknown chunking strategy: {strategy}")
