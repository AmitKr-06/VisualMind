"""
Hybrid retriever — BM25 + FAISS fused with RRF, then reranked.

Pipeline:
  1. BM25  → top-30 keywords
  2. FAISS → top-30 dense
  3. RRF   → fuse the two lists into one ranked list
  4. Rerank (optional) → cross-encoder over the fused top-20
  5. Return the final top-K

Config matches Exp 2 Cell 26 and Exp 3 Cell 17:
- pool_size: 30 for each retriever
- rrf_k: 60
- rerank_top_k: 20 (before reranking)
- final_k: 20
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

from src.exception import CustomException
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger("visualmind.retrieval.hybrid")


# ============================================================
# Constants
# ============================================================
DEFAULT_POOL_SIZE = 30
DEFAULT_RRF_K = 60
DEFAULT_RERANK_TOP_K = 20
DEFAULT_FINAL_K = 20


# ============================================================
# Hybrid retriever
# ============================================================
class HybridRetriever:
    """Fuses BM25 + FAISS with RRF, optionally reranks."""

    def __init__(
        self,
        bm25: BM25Retriever,
        faiss: FAISSRetriever,
        reranker: Optional[Reranker] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.bm25 = bm25
        self.faiss = faiss
        self.reranker = reranker
        self.pool_size = pool_size
        self.rrf_k = rrf_k

        logger.info(
            f"HybridRetriever ready | BM25={len(bm25.chunks)} chunks | "
            f"FAISS={faiss.index.ntotal} vectors | reranker={'on' if reranker else 'off'}"
        )

    # --------------------------------------------------------
    def _rrf_fuse(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion of the two ranked lists."""
        fused: Dict[str, Dict[str, Any]] = {}

        for lst in (dense, sparse):
            for r, item in enumerate(lst, start=1):
                cid = item["chunk_id"]
                entry = fused.setdefault(cid, {"item": item, "score": 0.0})
                entry["score"] += 1.0 / (self.rrf_k + r)

        ranked = sorted(fused.values(), key=lambda e: -e["score"])
        out = []
        for i, e in enumerate(ranked, start=1):
            item = dict(e["item"])
            item["rank"] = i
            item["rrf_score"] = round(e["score"], 6)
            out.append(item)
        return out

    # --------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: int = DEFAULT_FINAL_K,
        use_reranker: bool = True,
        use_rrf_rerank: bool = True,
        use_id_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline.

        Args:
            question: The user's question.
            k: Final number of results to return.
            use_reranker: Apply the cross-encoder reranker.
            use_rrf_rerank: Fuse reranker order with retriever order via RRF.
            use_id_boost: Move named IDs (Activity 11.2, etc.) to the front.

        Returns:
            Top-k chunks.
        """
        try:
            # 1. BM25
            sparse = self.bm25.search(question, k=self.pool_size)

            # 2. FAISS
            dense = self.faiss.search(question, k=self.pool_size)

            # 3. RRF fuse
            fused = self._rrf_fuse(dense, sparse)
            logger.debug(f"RRF fused {len(dense)} + {len(sparse)} → {len(fused)} candidates")

            # 4. Rerank (optional)
            if use_reranker and self.reranker is not None:
                top_for_rerank = fused[:DEFAULT_RERANK_TOP_K]
                reranked = self.reranker.rerank(
                    question,
                    top_for_rerank,
                    top_k=k,
                    use_rrf=use_rrf_rerank,
                    use_id_boost=use_id_boost,
                )
                return reranked

            # No reranker → return fused top-k directly
            return fused[:k]

        except Exception as e:
            raise CustomException(e, sys) from e