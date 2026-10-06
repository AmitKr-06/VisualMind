"""
Cross-encoder reranker.

Uses BAAI/bge-reranker-base (from Exp 3) to rescore the top-K candidates
from the hybrid retriever. Blends the reranker order with the retrieval
order via RRF, and applies an ID boost for questions that name an
Activity / Example / Table / Figure.

Config matches Exp 3 / Exp 9:
- model: BAAI/bge-reranker-base
- max_length: 512
- rrf_k: 60 (RRF fusion of retriever + reranker)
- ID boost: enabled (questions naming "Activity 11.2" etc.)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import CrossEncoder

from src.exception import CustomException

logger = logging.getLogger("visualmind.retrieval.reranker")


# ============================================================
# Constants
# ============================================================
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_LEN = 512
RRF_K = 60

# Matches "Activity 11.2", "Example 5.4", "Table 11.1", "Figure 5.9", "Fig 5.9"
ID_PAT = re.compile(r"\b(activity|example|table|figure|fig)\.?\s*(\d+\.\d+)", re.I)
# Matches a chunk that STARTS with that ID ("Activity 11.2 ..." at the top of the chunk)
HEAD_PAT = re.compile(r"^\s*(activity|example|table|figure|fig)\.?\s*(\d+\.\d+)", re.I)


def _kind(k: str) -> str:
    k = k.lower()
    return "figure" if k.startswith("fig") else k


def _rr_text(c: Dict[str, Any]) -> str:
    """Return the text used by the reranker (retrieval_text if present, else text)."""
    return c.get("retrieval_text") or c["text"]


def wanted_ids(question: str) -> set:
    """Extract (kind, number) pairs from the question: {('activity', '11.2'), ...}."""
    return {(_kind(k), n) for k, n in ID_PAT.findall(question)}


def head_ids(text: str) -> set:
    """Find (kind, number) pairs that START a paragraph in this chunk."""
    out = set()
    for para in text.split("\n\n"):
        m = HEAD_PAT.match(para.strip())
        if m:
            out.add((_kind(m.group(1)), m.group(2)))
    return out


def _id_boost(ranked: List[Dict[str, Any]], want: set) -> List[Dict[str, Any]]:
    """Move chunks whose heading matches the queried ID to the front."""
    if not want:
        return ranked
    front = [c for c in ranked if want & head_ids(c["text"])]
    kinds = {k for k, _ in want}
    own = [c for c in front if c["chunk_type"] in kinds]
    if own:
        own_ids = {c["chunk_id"] for c in own}
        front = own + [c for c in front if c["chunk_id"] not in own_ids]
    front_ids = {c["chunk_id"] for c in front}
    return front + [c for c in ranked if c["chunk_id"] not in front_ids]


# ============================================================
# Reranker
# ============================================================
class Reranker:
    """Cross-encoder reranker with RRF fusion and ID boost."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        max_length: int = RERANK_MAX_LEN,
        rrf_k: int = RRF_K,
        cache_dir: Optional[Path | str] = None,
    ):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading reranker: {model_name} on {device}")

            self.model = CrossEncoder(model_name, max_length=max_length, device=device)
            self.model_name = model_name
            self.max_length = max_length
            self.rrf_k = rrf_k

            self.cache_dir = Path(cache_dir) if cache_dir else None
            self._score_cache: Dict[str, float] = {}
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.cache_dir / "reranker_scores.json"
                if cache_file.exists():
                    self._score_cache = json.loads(cache_file.read_text(encoding="utf-8"))
                    logger.info(f"Loaded {len(self._score_cache)} cached scores")

            logger.info("  reranker loaded")

        except Exception as e:
            raise CustomException(e, sys) from e

    # --------------------------------------------------------
    def _key(self, q: str, text: str) -> str:
        return hashlib.md5((q + "||" + text).encode("utf-8")).hexdigest()

    def _save_cache(self):
        if not self.cache_dir:
            return
        (self.cache_dir / "reranker_scores.json").write_text(
            json.dumps(self._score_cache), encoding="utf-8"
        )

    def score(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        """Score (question, chunk) pairs. Cached on disk."""
        try:
            keys = [self._key(question, _rr_text(c)) for c in candidates]
            need = [(k, c) for k, c in zip(keys, candidates)
                    if k not in self._score_cache]

            if need:
                pairs = [(question, _rr_text(c)) for _, c in need]
                scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
                for (k, _), s in zip(need, scores):
                    self._score_cache[k] = float(s)
                self._save_cache()

            return [self._score_cache[k] for k in keys]

        except Exception as e:
            raise CustomException(e, sys) from e

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20,
        use_rrf: bool = True,
        use_id_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates.

        Args:
            question: The user's question.
            candidates: List of chunk dicts (with at least `chunk_id`, `text`).
            top_k: How many to return.
            use_rrf: Blend reranker order with retriever order (RRF).
            use_id_boost: Move chunks that match a named ID to the front.

        Returns:
            Reranked list of dicts, each with a `rerank_score` field.
        """
        try:
            if not candidates:
                return []

            scores = self.score(question, candidates)

            # 1. Sort by reranker score
            by_score = sorted(range(len(candidates)),
                              key=lambda i: (-scores[i], i))

            # 2. Optionally fuse with retrieval order via RRF
            if use_rrf:
                # retriever order = current candidate order (candidates come in retrieval rank order)
                ret_rank = {i: r for r, i in enumerate(range(len(candidates)), 1)}
                rer_rank = {i: r for r, i in enumerate(by_score, 1)}
                fused = {i: 1 / (self.rrf_k + ret_rank[i]) + 1 / (self.rrf_k + rer_rank[i])
                         for i in range(len(candidates))}
                order = sorted(range(len(candidates)),
                               key=lambda i: (-fused[i], i))
            else:
                order = by_score

            ranked = []
            for i in order:
                c = dict(candidates[i])
                c["rerank_score"] = round(scores[i], 4)
                ranked.append(c)

            # 3. Optional ID boost
            if use_id_boost:
                want = wanted_ids(question)
                ranked = _id_boost(ranked, want)

            # 4. Truncate
            ranked = ranked[:top_k]
            for i, c in enumerate(ranked, start=1):
                c["rank"] = i

            return ranked

        except Exception as e:
            raise CustomException(e, sys) from e