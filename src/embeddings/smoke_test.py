"""
Smoke test for the embeddings module.

Run from the project root:
    python -m src.embeddings.smoke_test
"""
import logging
import sys
from pathlib import Path

# Add project root to sys.path so "src.*" imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401
from src.embeddings.embedding import (
    embed_chunks,
    embed_query,
    load_embeddings,
    embedding_cache_path,
    DEFAULT_MODEL,
)

logger = logging.getLogger("visualmind.embeddings.smoke_test")


def main():
    logger.info("=" * 70)
    logger.info("EMBEDDINGS SMOKE TEST")
    logger.info("=" * 70)

    # 1. Locate the chunks file
    chunks_file = PROJECT_ROOT / "data" / "interim" / "chunks_smoke_test.jsonl"
    if not chunks_file.exists():
        logger.error(f"Missing {chunks_file}. Run the ingestion smoke test first.")
        sys.exit(1)

    logger.info(f"Chunks file: {chunks_file}")

    # 2. Embed all chunks
    logger.info("")
    logger.info("Embedding chunks...")
    result = embed_chunks(chunks_file, model_name=DEFAULT_MODEL)

    # 3. Summary
    logger.info("")
    logger.info("SUMMARY")
    logger.info("-" * 70)
    for k, v in result.summary().items():
        logger.info(f"  {k}: {v}")

    # 4. Verify norms
    norms = (result.vectors ** 2).sum(axis=1) ** 0.5
    logger.info("")
    logger.info("Vector checks:")
    logger.info(f"  min L2 norm : {norms.min():.4f}")
    logger.info(f"  max L2 norm : {norms.max():.4f}")
    logger.info(f"  mean L2 norm: {norms.mean():.4f}")
    assert abs(norms.mean() - 1.0) < 1e-3, "vectors should be ~unit norm"
    logger.info("  → all vectors are unit-length")

    # 5. Verify cache
    cached = load_embeddings(chunks_file, DEFAULT_MODEL)
    assert cached is not None, "cache should exist after embedding"
    assert (cached.vectors == result.vectors).all(), "cache must match fresh vectors"
    logger.info("  → cache round-trip verified")

    # 6. Query embedding — cosine similarity sanity check
    logger.info("")
    logger.info("Query sanity check:")
    q = "What is Ohm's law?"
    q_vec = embed_query(q)
    sims = result.vectors @ q_vec  # cosine similarity (both are unit-normed)
    top5 = sims.argsort()[::-1][:5]
    logger.info(f"  query: {q!r}")
    logger.info(f"  top-5 chunks by cosine similarity:")
    for rank, idx in enumerate(top5, 1):
        cid = result.chunk_ids[idx]
        logger.info(f"    [{rank}] {cid:<40s}  sim={sims[idx]:.4f}")

    # 7. Save location
    cache_path = embedding_cache_path(chunks_file, DEFAULT_MODEL)
    logger.info("")
    logger.info(f"Cache file: {cache_path}")
    logger.info(f"  exists: {cache_path.exists()}")
    logger.info(f"  size  : {cache_path.stat().st_size // 1024} KB")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EMBEDDINGS SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()