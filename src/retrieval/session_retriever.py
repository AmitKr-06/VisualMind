"""
Session retriever — build a HybridRetriever over in-memory chunks.

No disk writes. Embeddings are computed on the fly.
Used by the /upload endpoint to give each session its own index.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

import numpy as np

from src.exception import CustomException
from src.embeddings.embedding import embed_texts, DEFAULT_MODEL
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger("visualmind.retrieval.session")


def build_session_retriever(
    chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    with_reranker: bool = True,
) -> HybridRetriever:
    """
    Build a HybridRetriever from an in-memory list of chunks.

    Args:
        chunks: List of chunk dicts (each with chunk_id, text, ...).
        model_name: Embedding model.
        with_reranker: Include the cross-encoder.

    Returns:
        HybridRetriever ready to use.
    """
    try:
        if not chunks:
            raise ValueError("Cannot build a retriever from an empty chunk list")

        logger.info(f"Building session retriever over {len(chunks)} chunks...")

        # 1. BM25
        bm25 = BM25Retriever(chunks)

        # 2. Embed on the fly
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts, model_name=model_name, batch_size=32)
        chunk_ids = [c["chunk_id"] for c in chunks]

        # 3. FAISS
        faiss_retriever = FAISSRetriever(
            chunks=chunks,
            embeddings=vectors,
            chunk_ids=chunk_ids,
            model_name=model_name,
        )

        # 4. Reranker
        reranker = Reranker() if with_reranker else None

        # 5. Hybrid
        hybrid = HybridRetriever(bm25, faiss_retriever, reranker=reranker)
        logger.info(f"  session retriever ready ({len(chunks)} chunks)")
        return hybrid

    except Exception as e:
        raise CustomException(e, sys) from e