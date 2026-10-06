"""
Smoke test for the graphs module.

Run from the project root:
    python -m src.graphs.smoke_test
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

logger = logging.getLogger("visualmind.graphs.smoke_test")


def _print_final(label: str, out: dict):
    logger.info(f"  {label}:")
    logger.info(f"    status    : {out['status']}")
    logger.info(f"    answerable: {out['answerable']}")
    logger.info(f"    route     : {out['route']}")
    logger.info(f"    retries   : {out['retries']}")
    logger.info(f"    confidence: {out['confidence']}")
    logger.info(f"    citations : {out['citations'][:3]}{'...' if len(out['citations'])>3 else ''}")
    logger.info(f"    answer    : {out['answer'][:120]}{'...' if len(out['answer'])>120 else ''}")


def main():
    logger.info("=" * 70)
    logger.info("GRAPH SMOKE TEST")
    logger.info("=" * 70)

    # --- Build dependencies ---
    chunks_file = PROJECT_ROOT / "data" / "interim" / "chunks_smoke_test.jsonl"
    logger.info("")
    logger.info("Building dependencies...")
    retriever = get_retriever(chunks_file, with_reranker=False)
    manifest = load_figure_manifest(PROJECT_ROOT / "data" / "extracted")
    matcher = FigureMatcher(manifest)

    # --- Build the workflow ---
    logger.info("")
    logger.info("Building workflow...")
    wf = build_workflow(
        text_retriever=retriever,
        figure_matcher=matcher,
        image_analyzer=analyze_query_image,
    )

    # --- Test 1: text question ---
    logger.info("")
    logger.info("Test 1 — Text question: 'What is Ohm's law?'")
    out = wf.invoke(
        {"question": "What is Ohm's law?", "citations": [], "used_chunks": [],
         "used_figures": [], "notes": []},
        config={"configurable": {"thread_id": "smoke-1"}},
    )["final"]
    _print_final("text question", out)

    # --- Test 2: off-topic question ---
    logger.info("")
    logger.info("Test 2 — Off-topic question: 'Who won the cricket world cup in 2011?'")
    out = wf.invoke(
        {"question": "Who won the cricket world cup in 2011?",
         "citations": [], "used_chunks": [], "used_figures": [], "notes": []},
        config={"configurable": {"thread_id": "smoke-2"}},
    )["final"]
    _print_final("off-topic", out)

    # --- Test 3: multi-turn (same thread_id) ---
    logger.info("")
    logger.info("Test 3 — Multi-turn: reuse thread_id='smoke-3'")
    for q in ["What is Ohm's law?", "Why is it important?"]:
        out = wf.invoke(
            {"question": q, "citations": [], "used_chunks": [],
             "used_figures": [], "notes": []},
            config={"configurable": {"thread_id": "smoke-3"}},
        )["final"]
        logger.info(f"    Q: {q}")
        logger.info(f"      → status={out['status']} conf={out['confidence']} retries={out['retries']}")

    # --- Test 4: image-only ---
    logger.info("")
    logger.info("Test 4 — Image-only question")
    query_img = PROJECT_ROOT / "data" / "eval" / "images" / "test" / "in_5.1_phone_01.jpg"
    if not query_img.exists():
        candidates = list((PROJECT_ROOT / "data" / "eval" / "images" / "test").glob("in_*"))
        if candidates:
            query_img = candidates[0]

    if query_img.exists():
        out = wf.invoke(
            {"question": "", "image_path": str(query_img),
             "citations": [], "used_chunks": [], "used_figures": [], "notes": []},
            config={"configurable": {"thread_id": "smoke-4"}},
        )["final"]
        _print_final("image-only", out)
    else:
        logger.warning("  no test image found — skipped")

    logger.info("")
    logger.info("=" * 70)
    logger.info("GRAPH SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()