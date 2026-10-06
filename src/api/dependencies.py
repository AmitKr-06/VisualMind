"""
Lazy singletons — loaded once at FastAPI startup.

This is what makes requests fast: the retriever, matcher, and workflow
are built ONCE and reused across all requests.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import PROCESSED_DIR, RESULTS_DIR, EXTRACT_DIR, load_config

logger = logging.getLogger("visualmind.api.dependencies")


# ============================================================
# Config
# ============================================================
@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    logger.info("Loading frozen pipeline config...")
    cfg = load_config()
    logger.info("  config loaded")
    return cfg


# ============================================================
# Retriever
# ============================================================
@lru_cache(maxsize=1)
def get_retriever():
    from src.retrieval import get_retriever as _get
    cfg = get_config()
    frozen = cfg["frozen"]
    handoff = cfg["handoff"]
    chunk_file = PROCESSED_DIR / handoff["chunk_file"]
    logger.info(f"Building retriever from {chunk_file.name}")
    retriever = _get(chunk_file, with_reranker=True)
    logger.info("  retriever ready")
    return retriever


# ============================================================
# Figure matcher
# ============================================================
@lru_cache(maxsize=1)
def get_figure_matcher():
    from src.vision.figure_manifest import load_figure_manifest
    from src.vision.matchers import FigureMatcher
    logger.info("Loading figure manifest + CLIP matcher...")
    manifest = load_figure_manifest(EXTRACT_DIR)
    matcher = FigureMatcher(manifest)
    logger.info(f"  matcher ready ({len(manifest)} figures)")
    return matcher


# ============================================================
# Image analyzer
# ============================================================
@lru_cache(maxsize=1)
def get_image_analyzer():
    from src.vision.image_analyzer import analyze_query_image
    logger.info("Image analyzer ready")
    return analyze_query_image


# ============================================================
# Workflow (the compiled graph)
# ============================================================
@lru_cache(maxsize=1)
def get_workflow():
    from src.graphs import build_workflow
    cfg = get_config()
    frozen = cfg["frozen"]
    policy = frozen["policy"]

    logger.info("Building VisualMind workflow...")
    wf = build_workflow(
        text_retriever=get_retriever(),
        figure_matcher=get_figure_matcher(),
        image_analyzer=get_image_analyzer(),
        k_chunks=policy["k_chunks"],
        n_figures=policy["n_figures"],
        tau_text=policy["tau_text"],
    )
    logger.info("  workflow compiled")
    return wf


# ============================================================
# Metadata (for /health and /config)
# ============================================================
@lru_cache(maxsize=1)
def get_metadata() -> Dict[str, Any]:
    cfg = get_config()
    handoff = cfg["handoff"]
    frozen = cfg["frozen"]
    policy = frozen["policy"]
    return {
        "nodes": ["router", "retrieval", "answer", "reflection", "responder"],
        "chunk_file": handoff["chunk_file"],
        "policy": policy,
        "choice1": frozen["choice1"],
        "choice2": frozen["choice2"],
        "models": handoff["models"],
    }