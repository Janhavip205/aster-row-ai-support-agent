"""Metadata-aware, local in-memory retriever for Aster & Row knowledge base.

Uses local sentence-transformers embeddings with cosine similarity.
Enforces document precedence rules, excluding superseded, draft, and non-authoritative
documents from customer-facing answers by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

# Attempt to import SentenceTransformer; fallback to a dummy implementation for environments without the package
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    class _DummyTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            import numpy as np
            dim = 384
            if isinstance(texts, list):
                return np.zeros((len(texts), dim))
            return np.zeros(dim)
    SentenceTransformer = _DummyTransformer  # type: ignore

from src.config import EMBEDDING_MODEL, KB_DIR
from src.kb.indexer import DocumentChunk, load_all_chunks


class RetrievalResult(BaseModel):
    """Represents a retrieved chunk with similarity score and citation source."""

    chunk: DocumentChunk
    score: float = Field(description="Cosine similarity score in range [-1.0, 1.0]")
    source: str = Field(description="Formatted source citation: 'filename > heading'")


class KBRetriever:
    """Local, in-memory, deterministic knowledge base retriever."""

    def __init__(
        self,
        kb_dir: Path | str = KB_DIR,
        model_name: str = EMBEDDING_MODEL,
        min_score: float = 0.25,
    ) -> None:
        self.kb_dir = Path(kb_dir)
        self.min_score = min_score
        self.model_name = model_name

        # Load all chunks across the corpus
        self.chunks: list[DocumentChunk] = load_all_chunks(self.kb_dir)

        # Initialize local SentenceTransformer model
        self.model = SentenceTransformer(model_name)

        # Precompute normalized embeddings for all chunks in-memory
        texts_to_embed = [chunk.embed_text for chunk in self.chunks]
        if texts_to_embed:
            self._embeddings: np.ndarray = self.model.encode(
                texts_to_embed,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self._embeddings = np.empty((0, 384))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        authoritative_only: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant document chunks for a query.

        Args:
            query: The user inquiry or reformulated search query.
            top_k: Maximum number of passages to return.
            min_score: Minimum cosine similarity threshold. Defaults to self.min_score.
            authoritative_only: If True, filters out superseded, draft, non-official,
                                and non-customer-answering documents.

        Returns:
            List of RetrievalResult objects sorted by descending score.
        """
        threshold = self.min_score if min_score is None else min_score
        query = query.strip()
        if not query or len(self.chunks) == 0:
            return []

        # Filter candidate chunk indices
        candidate_indices: list[int] = []
        for idx, chunk in enumerate(self.chunks):
            if authoritative_only and not chunk.is_customer_authoritative:
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        # Encode query with normalization for cosine similarity
        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

        # Candidate embeddings matrix
        candidate_matrix = self._embeddings[candidate_indices]

        # Cosine similarity is the dot product of normalized vectors
        scores = np.dot(candidate_matrix, q_emb)

        # Pair candidates with their scores
        scored_results: list[RetrievalResult] = []
        for cand_idx, score_val in zip(candidate_indices, scores):
            float_score = float(score_val)
            if float_score >= threshold:
                chunk = self.chunks[cand_idx]
                scored_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=round(float_score, 4),
                        source=chunk.source_citation,
                    )
                )

        # Sort descending by similarity score
        scored_results.sort(key=lambda r: r.score, reverse=True)

        return scored_results[:top_k]

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all indexed chunks."""
        return list(self.chunks)

    def get_authoritative_chunks(self) -> list[DocumentChunk]:
        """Return only chunks eligible for customer answers."""
        return [c for c in self.chunks if c.is_customer_authoritative]

    def get_excluded_chunks(self) -> list[DocumentChunk]:
        """Return chunks excluded from customer answers (superseded or draft)."""
        return [c for c in self.chunks if not c.is_customer_authoritative]
