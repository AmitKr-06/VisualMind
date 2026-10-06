"""
FAISS dense retriever.

- Builds a FAISS IndexFlatIP (inner product = cosine similarity on unit vectors).
- Reads pre-computed embeddings from data/embeddings/<chunks>_<model>.npy
- Uses the BGE query prefix for the query embedding.
- Returns the same dict shape as BM25Retriever.

Config matches Exp 2 Cell 8-9:
- model: BAAI/bge-base-en-v1.5
- prefix: "Represent this sentence for searching relevant passages: "
- top_k: 30 by default
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.exception import CustomException
from src.embeddings.embedding import (
    embed_query,
    embed_chunks,
    load_embeddings,
    DEFAULT_MODEL,
)

logger = logging.getLogger("visualmind.retrieval.faiss")


# ============================================================
# Retriever
# ============================================================
class FAISSRetriever:
    """Dense retriever backed by FAISS IndexFlatIP."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        chunk_ids: List[str],
        model_name: str = DEFAULT_MODEL,
    ):
        """
        Args:
            chunks: List of chunk dicts (must match chunk_ids order).
            embeddings: (N, dim) float32 array, L2-normalized.
            chunk_ids: List of chunk ids in the SAME order as embeddings.
            model_name: Which embedding model to use for queries.
        """
        try:
            if not chunks:
                raise ValueError("FAISSRetriever requires a non-empty chunks list")
            if embeddings.shape[0] != len(chunk_ids):
                raise ValueError(
                    f"embeddings rows ({embeddings.shape[0]}) != chunk_ids ({len(chunk_ids)})"
                )
            if len(chunks) != len(chunk_ids):
                raise ValueError(
                    f"chunks ({len(chunks)}) != chunk_ids ({len(chunk_ids)})"
                )

            self.chunks = chunks
            self.embeddings = embeddings.astype("float32")
            self.chunk_ids = chunk_ids
            self.model_name = model_name
            self.dim = embeddings.shape[1]

            # Map id -> chunk for result assembly
            self.chunk_by_id = {c["chunk_id"]: c for c in chunks}

            # Build FAISS index
            logger.info(f"Building FAISS IndexFlatIP (dim={self.dim}, n={embeddings.shape[0]})...")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)

            logger.info(f"  FAISS index built with {self.index.ntotal} vectors")

        except Exception as e:
            raise CustomException(e, sys) from e

    def search(
        self,
        query: str,
        k: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index.

        Args:
            query: The user's question.
            k: How many results to return.

        Returns:
            List of dicts: {chunk_id, text, score, rank, source, page, ...}
        """
        try:
            q_vec = embed_query(query, model_name=self.model_name).reshape(1, -1)
            scores, indices = self.index.search(q_vec, k)

            results: List[Dict[str, Any]] = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx < 0:
                    continue
                cid = self.chunk_ids[int(idx)]
                c = self.chunk_by_id.get(cid)
                if c is None:
                    continue
                results.append({
                    **c,
                    "rank": rank,
                    "score": round(float(score), 4),
                    "retriever": "faiss",
                })
            return results

        except Exception as e:
            raise CustomException(e, sys) from e


# ============================================================
# Convenience: load chunks + embeddings from disk
# ============================================================
def build_faiss_from_jsonl(
    chunks_jsonl: Path | str,
    model_name: str = DEFAULT_MODEL,
    force_recompute: bool = False,
) -> FAISSRetriever:
    """
    Build a FAISSRetriever directly from a chunks JSONL file.
    Reuses (or builds) the embedding cache.
    """
    try:
        chunks_jsonl = Path(chunks_jsonl)
        if not chunks_jsonl.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl}")

        # Load chunks
        chunks: List[Dict[str, Any]] = []
        with open(chunks_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        logger.info(f"Loaded {len(chunks)} chunks from {chunks_jsonl}")

        # Embed (or load from cache)
        result = embed_chunks(chunks_jsonl, model_name=model_name,
                              force_recompute=force_recompute)

        return FAISSRetriever(
            chunks=chunks,
            embeddings=result.vectors,
            chunk_ids=result.chunk_ids,
            model_name=model_name,
        )

    except Exception as e:
        raise CustomException(e, sys) from e