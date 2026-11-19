"""Local grounded retrieval engine for NoticeWise (F4).

Features:
- Offline local TF-IDF / BM25 search over official knowledge base chunks.
- Zero external API calls, completely deterministic.
- Document-type filtering and affinity boosting for high precision.
- Returns top 3 to 4 authoritative chunks with exact source URLs, titles, and chunk IDs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import DocumentType, RetrievedChunk

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "kb_index.json"

_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_VECTORIZER: Optional[TfidfVectorizer] = None
_CHUNK_MATRIX: Optional[Any] = None


def load_kb_chunks() -> List[Dict[str, Any]]:
    """Load cached knowledge base chunks from data/kb_index.json."""
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    if not INDEX_PATH.exists():
        logger.warning("Knowledge base index not found at %s. Attempting to build...", INDEX_PATH)
        from scripts.build_index import build_kb_index
        build_kb_index()

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    _CHUNKS_CACHE = chunks
    return chunks


def _get_or_build_tfidf():
    """Build and cache the TF-IDF vectorizer and chunk embedding matrix."""
    global _VECTORIZER, _CHUNK_MATRIX
    if _VECTORIZER is not None and _CHUNK_MATRIX is not None:
        return _VECTORIZER, _CHUNK_MATRIX

    chunks = load_kb_chunks()
    corpus = [f"{c.get('title', '')} {c.get('content', '')}" for c in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    chunk_matrix = vectorizer.fit_transform(corpus)
    _VECTORIZER = vectorizer
    _CHUNK_MATRIX = chunk_matrix
    return _VECTORIZER, _CHUNK_MATRIX


def search_kb(
    query: str,
    doc_type: Optional[Union[str, DocumentType]] = None,
    top_k: int = 4,
) -> List[RetrievedChunk]:
    """Retrieve top-scoring knowledge base chunks relevant to query and notice doc_type.

    Args:
        query: Search query (e.g. extracted salient facts, issuer, penalty reason).
        doc_type: Optional document type filter/boost ('electricity', 'challan', 'tax', etc.).
        top_k: Number of chunks to return (default 4, maximum 4).

    Returns:
        List of RetrievedChunk models sorted by relevance score.
    """
    chunks = load_kb_chunks()
    if not chunks:
        logger.error("No knowledge base chunks available in %s", INDEX_PATH)
        return []

    top_k = min(max(1, top_k), 4)  # Constrain between 1 and 4

    vectorizer, chunk_matrix = _get_or_build_tfidf()

    # Clean and vectorize query
    cleaned_query = query.strip()
    if not cleaned_query:
        cleaned_query = "procedure grievance dispute complaint rules appeal"

    query_vec = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vec, chunk_matrix).flatten()

    # Normalise doc_type string
    type_str: Optional[str] = None
    if doc_type:
        type_str = doc_type.value if hasattr(doc_type, "value") else str(doc_type).lower()

    scored_chunks: List[RetrievedChunk] = []
    for idx, chunk in enumerate(chunks):
        base_score = float(similarities[idx])
        chunk_type = chunk.get("doc_type", "other")

        # Type affinity boost: reward chunks that match the classified notice category
        if type_str and chunk_type == type_str:
            final_score = base_score * 1.5 + 0.10
        elif type_str and chunk_type != "other" and chunk_type != type_str:
            # Penalize mismatching categories
            final_score = base_score * 0.4
        else:
            final_score = base_score

        scored_chunks.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                title=chunk["title"],
                source_url=chunk["source_url"],
                date_accessed=chunk.get("date_accessed", "2026-11-19"),
                content=chunk["content"],
                score=round(final_score, 4),
            )
        )

    # Sort descending by final score
    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    return scored_chunks[:top_k]


def get_top_retrieval_score(retrieved: List[RetrievedChunk]) -> float:
    """Return top similarity score from retrieved chunks, or 0.0 if empty."""
    if not retrieved:
        return 0.0
    return max(c.score for c in retrieved)
