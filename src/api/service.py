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

    Args:
        question: The user's question.
        thread_id: Optional checkpoint id for multi-turn.
        session_id: Reserved for module 10 (scoped retrieval).
        image_path: Optional image path.

    Returns:
        Final dict + elapsed_s + thread_id.
    """
    try:
        wf = get_workflow()
        tid = thread_id or f"api-{uuid.uuid4().hex[:8]}"

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