"""
Smoke test for the retrieval module.

Run from the project root:
    python -m src.retrieval.smoke_test
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401
from src.retrieval import get_retriever

logger = logging.getLogger("visualmind.retrieval.smoke_test")


def _show(results, label):
    logger.info(f"  {label}:")
    if not results:
        logger.info("    (no results)")
        return
    for r in results[:5]:
        logger.info(
            f"    [{r['rank']}] {r['chunk_id']:<40s} "
            f"score={r.get('score', r.get('rrf_score', 0)):.4f} "
            f"| {r.get('source', '?')[:20]:<20s} "
            f"| p{r.get('page', '?')}"
        )


def main():
    logger.info("=" * 70)
    logger.info("RETRIEVAL SMOKE TEST")
    logger.info("=" * 70)

    # 1. Locate chunks
    chunks_file = PROJECT_ROOT / "data" / "interim" / "chunks_smoke_test.jsonl"
    if not chunks_file.exists():
        logger.error(f"Missing {chunks_file}. Run ingestion first.")
        sys.exit(1)

    logger.info(f"Chunks file: {chunks_file}")

    # 2. Build retriever (BM25 + FAISS + RRF + reranker)
    logger.info("")
    logger.info("Building retriever...")
    retriever = get_retriever(chunks_file, with_reranker=True)

    # 3. Test questions
    queries = [
        "What is Ohm's law?",
        "How does a nephron filter blood?",
        "What is the purpose of potassium hydroxide in the CO2 activity?",
        "Activity 11.2: what apparatus is used?",   # ID-boost test
        "What is the capital of France?",           # out-of-scope test
    ]

    for q in queries:
        logger.info("")
        logger.info(f"Query: {q!r}")
        results = retriever.retrieve(q, k=5)
        _show(results, "hybrid + reranker (top-5)")

    # 4. Verify shape
    logger.info("")
    logger.info("Shape check:")
    r = retriever.retrieve("What is Ohm's law?", k=5)
    assert isinstance(r, list), "retrieve() must return a list"
    assert len(r) == 5, f"expected 5 results, got {len(r)}"
    assert all("chunk_id" in c and "text" in c for c in r), "chunks must have chunk_id and text"
    logger.info(f"  → returned {len(r)} chunks, all with chunk_id and text")

    logger.info("")
    logger.info("=" * 70)
    logger.info("RETRIEVAL SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()