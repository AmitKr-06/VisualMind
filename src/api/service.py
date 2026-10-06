"""
Business logic — wraps the workflow so routes stay thin.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from src.exception import CustomException
from src.api.dependencies import get_workflow

logger = logging.getLogger("visualmind.api.service")


# ============================================================
# Helpers
# ============================================================
def _empty_state(question: str, image_path: Optional[str], thread_id: str) -> Dict[str, Any]:
    return {
        "question":     question or "",
        "image_path":   image_path,
        "thread_id":    thread_id,
        "citations":    [],
        "used_chunks":  [],
        "used_figures": [],
        "notes":        [],
    }


# ============================================================
# Single question
# ============================================================
def ask(
    question: str,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the pipeline on one question.

    If session_id is provided AND that session has a retriever,
    the request is scoped to that session's chunks. Otherwise it falls
    back to the global workflow.
    """
    try:
        tid = thread_id or f"api-{uuid.uuid4().hex[:8]}"

        # --- Session-scoped path ---
        if session_id:
            from src.api.session_store import get_session
            sess = get_session(session_id)
            if sess is not None and sess.retriever is not None:
                from src.graphs import build_workflow
                from src.vision.image_analyzer import analyze_query_image
                from src.vision.figure_manifest import load_figure_manifest
                from src.vision.matchers import FigureMatcher
                from src.config import EXTRACT_DIR

                # Lightweight per-session workflow (figure matcher is optional)
                try:
                    manifest = load_figure_manifest(EXTRACT_DIR)
                    matcher = FigureMatcher(manifest)
                except Exception:
                    matcher = None

                wf = build_workflow(
                    text_retriever=sess.retriever,
                    figure_matcher=matcher,
                    image_analyzer=analyze_query_image,
                    k_chunks=3,
                    n_figures=2,
                    tau_text=0.115,
                    use_memory=False,
                )

                t0 = time.time()
                out = wf.invoke(
                    _empty_state(question, image_path, tid),
                    config={"configurable": {"thread_id": tid}},
                )
                elapsed = round(time.time() - t0, 2)
                final = dict(out.get("final") or {})
                final["elapsed_s"] = elapsed
                final["thread_id"] = tid
                final["session_id"] = session_id
                return final

        # --- Global path (fallback) ---
        wf = get_workflow()
        t0 = time.time()
        out = wf.invoke(
            _empty_state(question, image_path, tid),
            config={"configurable": {"thread_id": tid}},
        )
        elapsed = round(time.time() - t0, 2)
        final = dict(out.get("final") or {})
        final["elapsed_s"] = elapsed
        final["thread_id"] = tid
        return final

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Batch
# ============================================================
def ask_batch(questions: List[str]) -> Dict[str, Any]:
    """Run a list of questions sequentially, return each result + timing."""
    try:
        t0 = time.time()
        results = [ask(q) for q in questions]
        return {
            "results":   results,
            "count":     len(results),
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        raise CustomException(e, sys) from e