"""
Top-level retrieval factory.

Usage:
    from src.retrieval import get_retriever

    retriever = get_retriever(chunks_jsonl="data/interim/chunks_smoke_test.jsonl")
    results = retriever.retrieve("What is Ohm's law?", k=5)

By default, `get_retriever()` builds a full hybrid pipeline:
  BM25 + FAISS + RRF fusion + cross-encoder reranking.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.exception import CustomException
from src.retrieval.bm25_retriever import build_bm25_from_jsonl
from src.retrieval.faiss_retriever import build_faiss_from_jsonl
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.embeddings.embedding import DEFAULT_MODEL

logger = logging.getLogger("visualmind.retrieval")


# ============================================================
# Global cache — build once, reuse forever
# ============================================================
_RETRIEVER_CACHE: dict = {}


def get_retriever(
    chunks_jsonl: str | Path,
    model_name: str = DEFAULT_MODEL,
    with_reranker: bool = True,
    force_rebuild: bool = False,
) -> HybridRetriever:
    """
    Build (or return cached) HybridRetriever.

    Args:
        chunks_jsonl: Path to the chunks JSONL.
        model_name: Embedding model name.
        with_reranker: Include the cross-encoder reranker.
        force_rebuild: Ignore cache.

    Returns:
        A HybridRetriever ready to use.
    """
    try:
        chunks_jsonl = Path(chunks_jsonl)
        key = (str(chunks_jsonl), model_name, with_reranker)

        if not force_rebuild and key in _RETRIEVER_CACHE:
            logger.info(f"Using cached retriever for {chunks_jsonl.name}")
            return _RETRIEVER_CACHE[key]

        logger.info(f"Building HybridRetriever from {chunks_jsonl.name}")

        # 1. BM25
        bm25 = build_bm25_from_jsonl(chunks_jsonl)

        # 2. FAISS (loads/creates embeddings)
        faiss_retriever = build_faiss_from_jsonl(
            chunks_jsonl, model_name=model_name
        )

        # 3. Reranker (optional)
        reranker = Reranker() if with_reranker else None

        # 4. Hybrid
        hybrid = HybridRetriever(bm25, faiss_retriever, reranker=reranker)

        _RETRIEVER_CACHE[key] = hybrid
        return hybrid

    except Exception as e:
        raise CustomException(e, __import__("sys")) from e


def clear_cache():
    """Free the cached retriever (useful for tests)."""
    _RETRIEVER_CACHE.clear()
    logger.info("Retriever cache cleared")