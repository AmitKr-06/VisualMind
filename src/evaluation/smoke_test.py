"""
Smoke test for the evaluation module.

Runs a small subset (6 in-scope + 6 off-topic) so the test finishes fast.
Run from the project root:
    python -m src.evaluation.smoke_test
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401
from src.retrieval import get_retriever
from src.vision.figure_manifest import load_figure_manifest
from src.vision.matchers import FigureMatcher
from src.vision.image_analyzer import analyze_query_image
from src.graphs import build_workflow
from src.evaluation import run_questions, save_results, aggregate_metrics

logger = logging.getLogger("visualmind.evaluation.smoke_test")


def main():
    logger.info("=" * 70)
    logger.info("EVALUATION SMOKE TEST")
    logger.info("=" * 70)

    # --- Load held-out questions ---
    eval_dir = PROJECT_ROOT / "data" / "eval"
    test_in  = eval_dir / "gold_questions_test_v2.json"
    test_off = eval_dir / "gold_questions_offtopic_test_v1.json"

    if not test_in.exists():
        logger.error(f"Missing {test_in}")
        sys.exit(1)

    import json
    held_in  = json.loads(test_in.read_text(encoding="utf-8"))[:6]
    held_off = json.loads(test_off.read_text(encoding="utf-8"))[:6] if test_off.exists() else []

    logger.info(f"Loaded {len(held_in)} in-scope, {len(held_off)} off-topic questions")

    # --- Build the workflow ---
    logger.info("")
    logger.info("Building dependencies...")
    chunks_file = PROJECT_ROOT / "data" / "interim" / "chunks_smoke_test.jsonl"
    retriever = get_retriever(chunks_file, with_reranker=False)
    manifest  = load_figure_manifest(PROJECT_ROOT / "data" / "extracted")
    matcher   = FigureMatcher(manifest)

    logger.info("")
    logger.info("Building workflow...")
    wf = build_workflow(
        text_retriever=retriever,
        figure_matcher=matcher,
        image_analyzer=analyze_query_image,
    )

    # --- Run in-scope ---
    logger.info("")
    logger.info("Running in-scope questions...")
    results_in = run_questions(
        wf, held_in, kind="in_scope",
        thread_id_prefix="eval-in",
        verbose=True,
        text_retriever=retriever,
    )

    # --- Run off-topic ---
    results_off = []
    if held_off:
        logger.info("")
        logger.info("Running off-topic questions...")
        results_off = run_questions(
            wf, held_off, kind="offtopic",
            thread_id_prefix="eval-off",
            verbose=True,
        )

    # --- Aggregate ---
    logger.info("")
    logger.info("Aggregate metrics")
    metrics = aggregate_metrics(results_in, results_off)
    for k, v in metrics["in_scope"].items():
        logger.info(f"  in_scope/{k}: {v}")
    for k, v in metrics["offtopic"].items():
        logger.info(f"  offtopic/{k}: {v}")

    # --- Save ---
    out_dir = PROJECT_ROOT / "data" / "results"
    files = save_results(results_in, results_off, out_dir, run_version="smoke")
    logger.info("")
    logger.info("Saved files:")
    for name, p in files.items():
        logger.info(f"  {name}: {p}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()