"""Evaluation — run the pipeline on held-out sets and compute metrics."""

from .metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    citation_precision,
    refusal_rate,
    summarize_metrics,
)
from .runner import run_questions, EvalResult
from .report import (
    results_to_dataframe,
    save_results,
    aggregate_metrics,
)

__all__ = [
    "hit_at_k",
    "mean_reciprocal_rank",
    "citation_precision",
    "refusal_rate",
    "summarize_metrics",
    "run_questions",
    "EvalResult",
    "results_to_dataframe",
    "save_results",
    "aggregate_metrics",
]