"""
Metrics for the retrieval + answer pipeline.

Retrieval metrics (per question):
  - Hit@k: is the evidence chunk in the top-k results?
  - MRR:   1 / rank of the first evidence chunk
  - misses@k: evidence not found in top-k

Answer metrics (per question):
  - citation_precision: (# valid citations) / (# total citations)
  - answer_rate: status == "ok"
  - refusal_rate: status == "no_evidence" or "out_of_scope"
  - answerable_rate: answerable == True
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================
# Evidence matching
# ============================================================
def _norm(t: str) -> str:
    t = t.lower().replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"[\^_()]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_evidence(question: Dict[str, Any], text: str) -> bool:
    """Does a chunk's text contain ALL the gold must_contain phrases?"""
    t = _norm(text)
    must = question.get("must_contain", []) or []
    if not must:
        return False
    return all(_norm(p) in t for p in must)


def first_evidence_rank(question: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Optional[int]:
    """Rank (1-based) of the first chunk containing the full evidence."""
    doc = question.get("doc", "")
    for i, c in enumerate(chunks, start=1):
        if c.get("source") == doc and has_evidence(question, c.get("text", "")):
            return i
    return None


# ============================================================
# Retrieval metrics
# ============================================================
def hit_at_k(ranks: List[Optional[int]], k: int) -> float:
    """Fraction of questions where evidence is in top-k."""
    if not ranks:
        return 0.0
    return round(100 * sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 1)


def mean_reciprocal_rank(ranks: List[Optional[int]], cap: int = 5) -> float:
    """Mean of 1/rank. Misses contribute 0."""
    if not ranks:
        return 0.0
    vals = [1.0 / r if (r is not None and r <= cap) else 0.0 for r in ranks]
    return round(float(np.mean(vals)), 3)


def misses_at_k(ranks: List[Optional[int]], k: int) -> int:
    return sum(1 for r in ranks if r is None or r > k)


# ============================================================
# Answer metrics
# ============================================================
CITATION_RE = re.compile(r"\[([a-zA-Z0-9_\.]+)\]")


def citation_precision(citations: List[str], valid_ids: set) -> float:
    """
    Precision of the citations: (# valid) / (# total).
    Returns 1.0 if there are no citations (nothing wrong was said).
    """
    if not citations:
        return 1.0
    valid = sum(1 for c in citations if c in valid_ids)
    return round(valid / len(citations), 3)


def refusal_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results with status no_evidence or out_of_scope."""
    if not results:
        return 0.0
    refused = sum(1 for r in results if r.get("status") in ("no_evidence", "out_of_scope"))
    return round(100 * refused / len(results), 1)


def answer_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results with status == ok."""
    if not results:
        return 0.0
    ok = sum(1 for r in results if r.get("status") == "ok")
    return round(100 * ok / len(results), 1)


def answerable_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results that were marked answerable."""
    if not results:
        return 0.0
    yes = sum(1 for r in results if r.get("answerable") is True)
    return round(100 * yes / len(results), 1)


# ============================================================
# Aggregate
# ============================================================
def summarize_metrics(
    results: List[Dict[str, Any]],
    retrieval_ranks: Optional[List[Optional[int]]] = None,
) -> Dict[str, Any]:
    """
    Compute a dictionary of metrics from a list of pipeline results.

    Args:
        results: List of per-question result dicts.
        retrieval_ranks: Optional parallel list of ranks (for Hit@k, MRR).

    Returns:
        Dict of aggregate metrics.
    """
    out: Dict[str, Any] = {
        "n": len(results),
        "answer_rate_%": answer_rate(results),
        "refusal_rate_%": refusal_rate(results),
        "answerable_rate_%": answerable_rate(results),
        "mean_confidence": round(float(np.mean([r.get("confidence", 0.0) for r in results])), 3) if results else 0.0,
    }

    if retrieval_ranks:
        out["hit@1_%"] = hit_at_k(retrieval_ranks, 1)
        out["hit@3_%"] = hit_at_k(retrieval_ranks, 3)
        out["hit@5_%"] = hit_at_k(retrieval_ranks, 5)
        out["mrr"] = mean_reciprocal_rank(retrieval_ranks, cap=5)
        out["misses@5"] = misses_at_k(retrieval_ranks, 5)

    return out