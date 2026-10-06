"""
Evidence package builder.

Given a question and/or image, produces a strict evidence package:
    {
      "schema": "evidence_package_v1",
      "query":  {"question": ..., "image": ...},
      "status": "ok" | "no_evidence" | "out_of_scope",
      "chunks": [ {chunk_id, rank, role, text, source, page, chunk_type, figure_keys} ],
      "figures": [ {fig_key, rank, figure_id, doc, page, caption, description, file} ],
      "notes": [...]
    }

This is the EXACT shape the graph (module 7) will feed to the LLM.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exception import CustomException

logger = logging.getLogger("visualmind.multimodal.package")


# ============================================================
# Schema
# ============================================================
@dataclass
class EvidencePackage:
    """The unified evidence object passed to the LLM."""
    schema: str = "evidence_package_v1"
    query: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Validation
# ============================================================
REQUIRED_TOP_KEYS = {"schema", "query", "status", "chunks", "figures", "notes"}
ALLOWED_STATUSES = {"ok", "no_evidence", "out_of_scope"}


def validate_package(pkg: Dict[str, Any]) -> List[str]:
    """
    Check an evidence package for schema violations.

    Returns:
        List of error strings (empty = valid).
    """
    errors: List[str] = []

    if not isinstance(pkg, dict):
        return ["package must be a dict"]

    missing = REQUIRED_TOP_KEYS - set(pkg.keys())
    if missing:
        return [f"missing keys: {sorted(missing)}"]

    if pkg["status"] not in ALLOWED_STATUSES:
        errors.append(f"unknown status: {pkg['status']}")

    # Refused packages must be empty
    if pkg["status"] != "ok":
        if pkg["chunks"] or pkg["figures"]:
            errors.append("refused package must have empty chunks and figures")
        if not pkg["notes"]:
            errors.append("refused package must have a note explaining why")
        return errors

    # Answer chunks must be 1..3 (matches k_chunks policy)
    answer_chunks = [c for c in pkg["chunks"] if c.get("role") == "answer"]
    if not 1 <= len(answer_chunks) <= 3:
        errors.append(f"answer chunks must be 1..3, got {len(answer_chunks)}")

    # Chunk IDs must be unique
    ids = [c.get("chunk_id") for c in pkg["chunks"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chunk_id in package")

    # Each chunk must have required fields
    for c in pkg["chunks"]:
        for k in ("chunk_id", "text", "source", "page"):
            if k not in c:
                errors.append(f"chunk missing {k}")

    # Figures: at most 2, each with required fields
    if len(pkg["figures"]) > 2:
        errors.append(f"too many figures: {len(pkg['figures'])} > 2")

    for f in pkg["figures"]:
        for k in ("fig_key", "figure_id", "caption"):
            if k not in f:
                errors.append(f"figure missing {k}")

    return errors


# ============================================================
# Builder
# ============================================================
def _chunk_item(c: Dict[str, Any], rank: int, role: str) -> Dict[str, Any]:
    """Normalize a chunk dict for the package."""
    return {
        "chunk_id":   c.get("chunk_id"),
        "rank":       rank,
        "role":       role,
        "source":     c.get("source", ""),
        "page":       c.get("page", 0),
        "section":    c.get("section", ""),
        "chunk_type": c.get("chunk_type", "content"),
        "text":       c.get("text", ""),
        "figure_keys": c.get("figure_keys", []),
    }


def _figure_item(f: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Normalize a figure dict for the package."""
    return {
        "fig_key":     f.get("fig_key", ""),
        "rank":        rank,
        "figure_id":   f.get("figure_id", ""),
        "doc":         f.get("doc", ""),
        "page":        f.get("page", 0),
        "caption":     f.get("caption", ""),
        "description": f.get("description", ""),
        "file":        f.get("file", ""),
    }


def build_evidence_package(
    question: Optional[str] = None,
    image_path: Optional[str] = None,
    *,
    text_retriever=None,
    figure_matcher=None,
    image_analyzer=None,
    k_chunks: int = 3,
    n_figures: int = 2,
    tau_text: float = 0.115,
    out_of_scope: bool = False,
) -> Dict[str, Any]:
    """
    Build an evidence package.

    Args:
        question: Optional text question.
        image_path: Optional path to an image.
        text_retriever: A HybridRetriever instance (from module 3). Required.
        figure_matcher: A FigureMatcher instance (from module 5). Optional.
        image_analyzer: A callable analyze_query_image(path) (from module 5). Optional.
        k_chunks: How many answer chunks to attach.
        n_figures: How many figures to attach.
        tau_text: Threshold below which we return no_evidence.
        out_of_scope: If True, skip retrieval and return out_of_scope.

    Returns:
        Dict (the package)
    """
    try:
        # --- Out of scope fast-path ---
        if out_of_scope:
            return {
                "schema": "evidence_package_v1",
                "query": {"question": question, "image": image_path},
                "status": "out_of_scope",
                "chunks": [],
                "figures": [],
                "notes": ["input marked out_of_scope"],
            }

        if text_retriever is None:
            raise ValueError("text_retriever is required")

        # --- Image-only or mixed: analyze the image first ---
        query_text = question or ""
        image_analysis: Optional[Dict[str, Any]] = None

        if image_path and image_analyzer is not None:
            image_analysis = image_analyzer(image_path)

            if image_analysis is None:
                return {
                    "schema": "evidence_package_v1",
                    "query": {"question": question, "image": image_path},
                    "status": "out_of_scope",
                    "chunks": [],
                    "figures": [],
                    "notes": ["vision model returned no usable answer"],
                }

            if image_analysis.get("in_scope") is False:
                return {
                    "schema": "evidence_package_v1",
                    "query": {"question": question, "image": image_path},
                    "status": "out_of_scope",
                    "chunks": [],
                    "figures": [],
                    "notes": [
                        f"image refused: {image_analysis.get('reason', 'out of scope')}"
                    ],
                }

            # If no text question, use the image's description as the query
            if not query_text:
                query_text = image_analysis.get("description", "")
                if not query_text:
                    query_text = "diagram"

        # --- Retrieve chunks ---
        chunks = text_retriever.retrieve(query_text, k=20)

        if not chunks:
            return {
                "schema": "evidence_package_v1",
                "query": {"question": question, "image": image_path},
                "status": "no_evidence",
                "chunks": [],
                "figures": [],
                "notes": ["retriever returned zero chunks"],
            }

        # --- Check evidence strength ---
        top_score = max(c.get("score", 0.0) for c in chunks[:5])
        if top_score < tau_text:
            return {
                "schema": "evidence_package_v1",
                "query": {"question": question, "image": image_path},
                "status": "no_evidence",
                "chunks": [],
                "figures": [],
                "notes": [
                    f"top chunk score {top_score:.3f} < tau_text {tau_text:.3f}"
                ],
            }

        # --- Figures (optional) ---
        figures: List[Dict[str, Any]] = []
        if figure_matcher is not None:
            try:
                figures = figure_matcher.rank_by_text(query_text, k=n_figures * 2)
            except Exception as e:
                logger.warning(f"Figure matching failed: {e}")
                figures = []

        # --- Assemble package ---
        pkg = {
            "schema": "evidence_package_v1",
            "query": {"question": question, "image": image_path},
            "status": "ok",
            "chunks":  [_chunk_item(c, i, "answer") for i, c in enumerate(chunks[:k_chunks], 1)],
            "figures": [_figure_item(f, i) for i, f in enumerate(figures[:n_figures], 1)],
            "notes":   [],
        }

        # --- Validate ---
        errs = validate_package(pkg)
        if errs:
            logger.warning(f"Package validation warnings: {errs}")
            pkg["notes"].extend(errs)

        return pkg

    except Exception as e:
        raise CustomException(e, sys) from e