"""
BM25 keyword retriever.

Uses rank_bm25.BM25Okapi over tokenized chunk texts.
Falls back to exact-match search when the dense retriever misses.

Config matches Exp 2 Cell 26:
- tokenizer: lowercase, keep digits as single tokens (so "11.6" stays intact)
- stopwords: same list as Exp 2
- top_k: 30 by default (matches the fusion pool size)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.exception import CustomException

logger = logging.getLogger("visualmind.retrieval.bm25")


# ============================================================
# Constants
# ============================================================
STOPWORDS = {
    "what", "is", "the", "of", "a", "an", "how", "does", "do", "are",
    "in", "to", "and", "for", "why", "which", "who", "after", "whom",
    "besides", "it", "its", "be", "by", "from", "on", "or", "that",
    "this", "where",
}


# ============================================================
# Tokenizer
# ============================================================
def tokenize(text: str) -> List[str]:
    """
    Tokenize for BM25. Lowercase, keep decimals whole ("11.6"),
    drop common English stopwords.
    """
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+\d*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ============================================================
# Retriever
# ============================================================
class BM25Retriever:
    """BM25 retriever over a list of chunks."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        text_key: str = "text",
        id_key: str = "chunk_id",
    ):
        """
        Args:
            chunks: List of dicts (each with chunk_id, text, source, page, ...).
            text_key: Which dict key holds the searchable text.
            id_key: Which dict key holds the chunk id.
        """
        try:
            if not chunks:
                raise ValueError("BM25Retriever requires a non-empty chunks list")

            self.chunks = chunks
            self.text_key = text_key
            self.id_key = id_key

            logger.info(f"Building BM25 index over {len(chunks)} chunks...")
            tokenized = [tokenize(c[text_key]) for c in chunks]
            self.bm25 = BM25Okapi(tokenized)
            self.chunk_by_id = {c[id_key]: c for c in chunks}

            logger.info(f"  BM25 index built ({len(chunks)} docs)")

        except Exception as e:
            raise CustomException(e, sys) from e

    def search(
        self,
        query: str,
        k: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index.

        Args:
            query: The user's question.
            k: How many results to return.

        Returns:
            List of dicts: {chunk_id, text, score, rank, source, page, ...}
        """
        try:
            scores = self.bm25.get_scores(tokenize(query))
            order = np.argsort(-scores)[:k]

            results: List[Dict[str, Any]] = []
            for rank, idx in enumerate(order, start=1):
                c = self.chunks[int(idx)]
                results.append({
                    **c,
                    "rank": rank,
                    "score": round(float(scores[int(idx)]), 4),
                    "retriever": "bm25",
                })
            return results

        except Exception as e:
            raise CustomException(e, sys) from e


# ============================================================
# Convenience: load chunks from JSONL
# ============================================================
def build_bm25_from_jsonl(
    chunks_jsonl: Path | str,
) -> BM25Retriever:
    """
    Build a BM25Retriever directly from a chunks JSONL file.
    """
    try:
        chunks_jsonl = Path(chunks_jsonl)
        if not chunks_jsonl.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl}")

        chunks: List[Dict[str, Any]] = []
        with open(chunks_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        logger.info(f"Loaded {len(chunks)} chunks from {chunks_jsonl}")
        return BM25Retriever(chunks)

    except Exception as e:
        raise CustomException(e, sys) from e