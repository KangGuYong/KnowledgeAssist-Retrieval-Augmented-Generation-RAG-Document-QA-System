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


def _merge_group(group: List[Document]) -> Document:
    """Fold a run of page Documents into one, keeping their provenance.

    The first page's metadata wins so citations still point at where the
    chunk starts, but anything the source view needs from the later pages -
    their images, their OCR flags - has to be carried across too, or a
    merged chunk would silently lose the figures it quotes.
    """
    metadata = dict(group[0].metadata)

    if "page_number" in group[-1].metadata:
        metadata["page_number_end"] = group[-1].metadata["page_number"]

    if any("image_ids" in doc.metadata for doc in group):
        image_ids: List[str] = []
        for doc in group:
            for image_id in (doc.metadata.get("image_ids") or "").split(","):
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
        metadata["image_ids"] = ",".join(image_ids)

    if any("ocr_used" in doc.metadata for doc in group):
        metadata["ocr_used"] = any(doc.metadata.get("ocr_used") for doc in group)
    if any("ocr_image_count" in doc.metadata for doc in group):
        metadata["ocr_image_count"] = sum(doc.metadata.get("ocr_image_count") or 0 for doc in group)
    if any("full_page_ocr" in doc.metadata for doc in group):
        metadata["full_page_ocr"] = any(doc.metadata.get("full_page_ocr") for doc in group)

    return Document(
        page_content="\n\n".join(doc.page_content for doc in group), metadata=metadata
    )


def merge_pages(documents: List[Document], target_size: int) -> List[Document]:
    """Join consecutive pages of the same file until they reach target_size.

    Both splitters work one Document at a time and never merge across them,
    while load_pdf() emits one Document per page. On a large-print PDF
    holding ~200 characters per page that capped every chunk at a fraction of
    chunk_size, and left the semantic percentile to be computed over the two
    or three sentence distances one page happened to contain. Merging first
    means a chunk is bounded by the requested size rather than by the
    document's page layout.
    """
    merged: List[Document] = []
    group: List[Document] = []

    for doc in documents:
        if group and doc.metadata.get("source") != group[0].metadata.get("source"):
            merged.append(_merge_group(group))
            group = []

        group.append(doc)
        # "\n\n" between pages, matching how _merge_group joins them.
        size = sum(len(d.page_content) for d in group) + 2 * (len(group) - 1)
        if size >= target_size:
            merged.append(_merge_group(group))
            group = []

    if group:
        merged.append(_merge_group(group))

    return merged


class SemanticChunker:
    """Splits text into chunks at embedding-similarity breakpoints.

    Any distance above the given percentile of all consecutive-sentence
    distances is treated as a breakpoint (notebook cell 102) - the notebook's
    only tunable parameter.

    min_chunk_chars is this port's one addition, defaulting to 0 so the
    notebook's behaviour is what you get unless you ask for otherwise: a
    breakpoint that would emit a chunk shorter than it is carried into the
    next chunk instead. Without it a run of short lines produces chunks too
    small to answer anything - the worst offenders in practice being chunks
    holding nothing but a page number.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        breakpoint_percentile: float = 95.0,
        min_chunk_chars: int = 0,
    ):
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_chars = min_chunk_chars

    def _sentences(self, text: str) -> List[dict]:
        single_sentences = _SENTENCE_SPLIT_RE.split(text)
        sentences = [{"sentence": s, "index": i} for i, s in enumerate(single_sentences)]
        return _combine_sentences(sentences, self.buffer_size)

    def _chunks_from(self, sentences: List[dict], distances: List[float], threshold: float) -> List[str]:
        chunks: List[str] = []
        start = 0
        for i, distance in enumerate(distances):
            if distance <= threshold:
                continue
            candidate = " ".join(s["sentence"] for s in sentences[start:i + 1])
            if len(candidate) < self.min_chunk_chars:
                continue  # too small to stand alone - keep reading past this breakpoint
            chunks.append(candidate)
            start = i + 1

        if start < len(sentences):
            tail = " ".join(s["sentence"] for s in sentences[start:])
            # A leftover tail below the minimum belongs to the chunk before it;
            # there is no following chunk to carry it into.
            if chunks and len(tail) < self.min_chunk_chars:
                chunks[-1] = chunks[-1] + " " + tail
            else:
                chunks.append(tail)
        return chunks

    def split_text(self, text: str) -> List[str]:
        sentences = self._sentences(text)
        if len(sentences) <= 1:
            return [s["sentence"] for s in sentences]

        embeddings = self.embeddings.embed_documents(
            [s["combined_sentence"] for s in sentences]
        )
        distances = _cosine_distances(embeddings)
        if not distances:
            return [s["sentence"] for s in sentences]

        return self._chunks_from(sentences, distances, np.percentile(distances, self.breakpoint_percentile))

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents, carrying each parent's metadata onto every chunk.

        The breakpoint threshold is a percentile, i.e. a relative measure, so
        it is computed once over every document's distances rather than per
        document. Computing it per document forces roughly the same number of
        breaks into each one no matter what it contains, which on short
        documents means splitting text that has no topic shift at all.
        Chunks still never span two documents - only the threshold is shared.
        """
        per_document = [(doc, self._sentences(doc.page_content)) for doc in documents]
        combined = [s["combined_sentence"] for _, sentences in per_document for s in sentences]
        if not combined:
            return []

        embeddings = self.embeddings.embed_documents(combined)

        offset = 0
        per_document_distances = []
        for _, sentences in per_document:
            per_document_distances.append(_cosine_distances(embeddings[offset:offset + len(sentences)]))
            offset += len(sentences)

        all_distances = [d for distances in per_document_distances for d in distances]
        threshold = (
            np.percentile(all_distances, self.breakpoint_percentile)
            if all_distances
            else float("inf")
        )

        result = []
        for (doc, sentences), distances in zip(per_document, per_document_distances):
            for chunk_text in self._chunks_from(sentences, distances, threshold):
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
            min_chunk_chars=settings.semantic_chunker_min_chunk_chars,
        )

    raise ValueError(f"Unknown chunking strategy: {strategy}")
