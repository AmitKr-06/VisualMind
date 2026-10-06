"""
Aggregate metrics and save the evaluation report.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.exception import CustomException
from src.evaluation.metrics import summarize_metrics
from src.evaluation.runner import EvalResult

logger = logging.getLogger("visualmind.evaluation.report")


# ============================================================
# Convert to DataFrame
# ============================================================
def results_to_dataframe(results: List[EvalResult]) -> pd.DataFrame:
    """Convert EvalResult list into a pandas DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


# ============================================================
# Aggregate
# ============================================================
def aggregate_metrics(
    in_scope: List[EvalResult],
    offtopic: List[EvalResult],
) -> Dict[str, Any]:
    """Compute the full metric dictionary for a run."""
    in_ranks  = [r.retrieval_rank for r in in_scope]
    in_dicts  = [r.to_dict() for r in in_scope]
    off_dicts = [r.to_dict() for r in offtopic]

    return {
        "in_scope": summarize_metrics(in_dicts, retrieval_ranks=in_ranks),
        "offtopic": summarize_metrics(off_dicts),
    }


# ============================================================
# Save
# ============================================================
def save_results(
    in_scope: List[EvalResult],
    offtopic: List[EvalResult],
    output_dir: Path | str,
    run_version: str = "v1",
) -> Dict[str, Path]:
    """
    Save per-question CSVs + a metrics JSON.

    Returns:
        Dict of {name: path} for the files written.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files: Dict[str, Path] = {}

        # Per-question CSVs
        df_in  = results_to_dataframe(in_scope)
        df_off = results_to_dataframe(offtopic)

        path_in = output_dir / f"eval_{run_version}_in_scope.csv"
        path_off = output_dir / f"eval_{run_version}_offtopic.csv"
        df_in.to_csv(path_in, index=False)
        df_off.to_csv(path_off, index=False)
        files["in_scope_csv"]  = path_in
        files["offtopic_csv"]  = path_off

        # Metrics JSON
        metrics = aggregate_metrics(in_scope, offtopic)
        metrics["run_version"] = run_version
        metrics["n_in_scope"]  = len(in_scope)
        metrics["n_offtopic"]  = len(offtopic)

        path_metrics = output_dir / f"eval_{run_version}_metrics.json"
        path_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        files["metrics_json"] = path_metrics

        logger.info(f"Saved evaluation report to {output_dir}")
        for name, p in files.items():
            logger.info(f"  {name}: {p.name}")

        return files

    except Exception as e:
        raise CustomException(e, sys) from e