"""
Run the workflow over a list of questions and collect per-question results.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.exception import CustomException
from src.evaluation.metrics import first_evidence_rank

logger = logging.getLogger("visualmind.evaluation.runner")


# ============================================================
# Data structure
# ============================================================
@dataclass
class EvalResult:
    question: str
    kind: str = "in_scope"        # "in_scope" | "offtopic"
    gold_type: str = ""
    status: str = ""
    answerable: bool = False
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)
    answer_len: int = 0
    retries: int = 0
    route: str = ""
    elapsed_s: float = 0.0
    retrieval_rank: Optional[int] = None    # rank of the gold evidence chunk

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Runner
# ============================================================
def run_questions(
    workflow,
    questions: List[Dict[str, Any]],
    kind: str = "in_scope",
    thread_id_prefix: str = "eval",
    verbose: bool = True,
    text_retriever=None,
) -> List[EvalResult]:
    """
    Run every question through the workflow.

    Args:
        workflow: Compiled LangGraph workflow.
        questions: List of dicts with at least `question` and `doc`.
        kind: "in_scope" or "offtopic".
        thread_id_prefix: Prefix for thread IDs (must be unique per Q).
        verbose: Print progress.
        text_retriever: Optional — if provided, compute retrieval_rank per Q.

    Returns:
        List of EvalResult.
    """
    try:
        results: List[EvalResult] = []
        n = len(questions)
        t_start = time.time()

        for i, q in enumerate(questions, start=1):
            question_text = q.get("question", "")
            tid = f"{thread_id_prefix}-{uuid.uuid4().hex[:8]}"

            t0 = time.time()
            try:
                out = workflow.invoke(
                    {
                        "question":     question_text,
                        "image_path":   None,
                        "thread_id":    tid,
                        "citations":    [],
                        "used_chunks":  [],
                        "used_figures": [],
                        "notes":        [],
                    },
                    config={"configurable": {"thread_id": tid}},
                )["final"]
                elapsed = round(time.time() - t0, 2)
            except Exception as e:
                logger.warning(f"Q{i} failed: {e}")
                out = {
                    "status": "error", "answer": "", "citations": [],
                    "confidence": 0.0, "answerable": False, "retries": 0,
                    "route": "text",
                }
                elapsed = round(time.time() - t0, 2)

            # Retrieval rank (optional)
            rank: Optional[int] = None
            if text_retriever is not None and kind == "in_scope":
                try:
                    chunks = text_retriever.retrieve(question_text, k=20)
                    rank = first_evidence_rank(q, chunks)
                except Exception as e:
                    logger.warning(f"Retrieval rank failed for Q{i}: {e}")

            res = EvalResult(
                question=question_text[:100],
                kind=kind,
                gold_type=q.get("type", ""),
                status=out.get("status", ""),
                answerable=bool(out.get("answerable", False)),
                confidence=float(out.get("confidence", 0.0)),
                citations=list(out.get("citations", [])),
                answer_len=len(out.get("answer", "")),
                retries=int(out.get("retries", 0)),
                route=out.get("route", "text"),
                elapsed_s=elapsed,
                retrieval_rank=rank,
            )
            results.append(res)

            if verbose:
                logger.info(
                    f"  [{i:>2}/{n}] {res.status:<18s} "
                    f"conf={res.confidence:.2f} retries={res.retries} "
                    f"rank={res.retrieval_rank} | {question_text[:60]}"
                )

        total = round(time.time() - t_start, 1)
        logger.info(f"Ran {n} questions in {total}s ({total/n:.1f}s each)")
        return results

    except Exception as e:
        raise CustomException(e, sys) from e