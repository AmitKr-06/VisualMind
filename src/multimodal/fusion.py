"""
Fusion — combine multiple ranked lists into one.

Used for:
- Chunks: fuse BM25 + FAISS (already in retrieval.hybrid, exposed here for reuse)
- Figures: fuse CLIP image + CLIP text (from vision.matchers)
- Cross: fuse figure ranking with chunk ranking by RRF when relevant
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

from src.exception import CustomException

logger = logging.getLogger("visualmind.multimodal.fusion")


# ============================================================
# Constants
# ============================================================
DEFAULT_RRF_K = 60


# ============================================================
# RRF (Reciprocal Rank Fusion)
# ============================================================
def fuse_rrf(
    ranked_lists: List[List[Dict[str, Any]]],
    id_key: str,
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists with RRF.

    Args:
        ranked_lists: List of ranked lists, each containing dicts with `id_key`.
        id_key: The dict key that identifies items uniquely (e.g. "chunk_id").
        k: RRF constant (default 60).
        top_k: Optional truncation after fusion.

    Returns:
        Fused ranked list, each item with an additional "rrf_score" and "rank".
    """
    try:
        fused: Dict[str, Dict[str, Any]] = {}

        for lst in ranked_lists:
            for r, item in enumerate(lst, start=1):
                key = item.get(id_key)
                if key is None:
                    continue
                entry = fused.setdefault(key, {"item": item, "score": 0.0})
                entry["score"] += 1.0 / (k + r)

        ranked = sorted(fused.values(), key=lambda e: -e["score"])

        out: List[Dict[str, Any]] = []
        for i, e in enumerate(ranked, start=1):
            item = dict(e["item"])
            item["rank"] = i
            item["rrf_score"] = round(e["score"], 6)
            out.append(item)

        if top_k:
            out = out[:top_k]
        return out

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Convenience wrappers
# ============================================================
def fuse_chunks_and_figures(
    chunks: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    top_k_chunks: int = 3,
    top_k_figures: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Truncate chunk + figure lists to their final size.

    No cross-fusion here — chunks and figures answer different
    parts of the question, so we keep both lists.

    Args:
        chunks: Fused chunk results (already ranked).
        figures: Fused figure results (already ranked).
        top_k_chunks: How many chunks to keep.
        top_k_figures: How many figures to keep.

    Returns:
        {"chunks": [...], "figures": [...]}
    """
    try:
        return {
            "chunks":  chunks[:top_k_chunks],
            "figures": figures[:top_k_figures],
        }
    except Exception as e:
        raise CustomException(e, sys) from e